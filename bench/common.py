"""ComfyUI-as-a-library plumbing for the quant benchmark.

Imported by prep.py and run.py inside a bench container whose /workspace is
the comfyui checkout + venv (see README). Nothing here touches the live
ComfyUI server: it is a separate process with ComfyUI's default flags (no
--fast, no sage, no --highvram), so the numbers describe stock ComfyUI.
"""

import contextlib
import gc
import json
import os
import sys
import time

COMFY = os.environ.get("COMFY_DIR", "/workspace/ComfyUI")
if COMFY not in sys.path:
    sys.path.insert(0, COMFY)

# QBENCH_DYNAMIC=1 turns on ComfyUI's dynamic VRAM (comfy-aimdo), the loader
# stock ComfyUI uses by default on NVIDIA: weights that do not fit stay in
# system RAM and stream to the GPU as needed. fit.py runs this way, so a
# model bigger than the card runs the way it would for a user. run.py keeps
# the static loader (everything resident), which the full-quality pass
# measures. Set up in the same order as ComfyUI's main.py: init before torch,
# devices after comfy.model_management.
DYNAMIC = os.environ.get("QBENCH_DYNAMIC") == "1"
if os.name == "nt":
    os.environ.setdefault("MIMALLOC_PURGE_DELAY", "0")   # as ComfyUI's main.py does on Windows
if DYNAMIC:
    from comfy.cli_args import args as comfy_args  # noqa: E402
    # Test knob: extra VRAM for dynamic VRAM to leave free, to make a big card
    # behave like a smaller one when checking the offload path.
    if os.environ.get("QBENCH_VRAM_HEADROOM"):
        comfy_args.vram_headroom = float(os.environ["QBENCH_VRAM_HEADROOM"])
    # QBENCH_FAST_DISK=1: the offload policy ComfyUI picks for a model on a
    # local NVMe drive (stream from disk). Inside WSL2 the disk is a virtual
    # SCSI device, so ComfyUI's detection says "slow" there; native Windows on
    # the same NVMe says "fast". Set this on WSL2 hosts to measure what a
    # Windows user gets.
    if os.environ.get("QBENCH_FAST_DISK") == "1":
        comfy_args.fast_disk = True
    import comfy_aimdo.control  # noqa: E402
    comfy_aimdo.control.init(simple_vram_headroom=None, nvml_pressure=not comfy_args.disable_nvml_pressure)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import comfy.model_management as mm  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.sample  # noqa: E402
import comfy.samplers  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import comfy.ldm.modules.attention as attn_mod  # noqa: E402

comfy.utils.PROGRESS_BAR_ENABLED = False

if DYNAMIC:
    import comfy.memory_management  # noqa: E402
    import comfy.model_patcher  # noqa: E402
    _devs = mm.get_all_torch_devices()
    try:
        _ok = comfy_aimdo.control.init_devices((d.index, int(comfy_args.vram_headroom * 1024 ** 3)) for d in _devs)
    except TypeError:   # comfy-aimdo 0.4.9 protocol
        _ok = comfy_aimdo.control.init_devices(d.index for d in _devs)
    if not _ok:
        raise RuntimeError("QBENCH_DYNAMIC=1 but comfy-aimdo did not initialize")
    comfy.model_patcher.CoreModelPatcher = comfy.model_patcher.ModelPatcherDynamic
    comfy.memory_management.aimdo_enabled = True


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def free_all():
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache(True)
    torch.cuda.empty_cache()


# ── text encoders / conditioning ────────────────────────────────────────
def load_te(path, typ):
    clip_type = getattr(comfy.sd.CLIPType, typ.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
    return comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=None, clip_type=clip_type, model_options={})


def encode(clip, text):
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))


# ── VAE ─────────────────────────────────────────────────────────────────
def load_vae(path):
    sd, meta = comfy.utils.load_torch_file(path, return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, metadata=meta)
    vae.throw_exception_if_invalid()
    return vae


def decode_png(vae, samples, path):
    from PIL import Image
    import numpy as np
    img = vae.decode(samples)
    if img.ndim == 5:
        img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
    arr = (img[0].clamp(0, 1).cpu().float().numpy() * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(arr).save(path, compress_level=4)


# ── diffusion model ─────────────────────────────────────────────────────
@contextlib.contextmanager
def emulated(fmt):
    """Run `fmt` without its native kernels (dequant → bf16 matmul), the way
    ComfyUI runs it on a GPU that lacks them. ComfyUI decides this per GPU in
    comfy.ops.get_disabled_quant_formats, both at load and per forward."""
    if not fmt:
        yield
        return
    orig = comfy.ops.get_disabled_quant_formats
    comfy.ops.get_disabled_quant_formats = lambda device=None: set(orig(device)) | {fmt}
    try:
        yield
    finally:
        comfy.ops.get_disabled_quant_formats = orig


def load_diffusion(path, load_dtype=None):
    opts = {}
    if load_dtype == "fp8_e4m3fn":
        opts["dtype"] = torch.float8_e4m3fn
    # Static (non-dynamic) patcher unless QBENCH_DYNAMIC: weights resident, so
    # the VRAM split into weights vs activations is measurable and timing has
    # no streaming in it.
    return comfy.sd.load_diffusion_model(path, model_options=opts, disable_dynamic=not DYNAMIC)


def apply_lora(model, path, strength):
    lora = comfy.utils.load_torch_file(path, safe_load=True)
    model, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0)
    return model


def aura_shift(model, shift):
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
    return ModelSamplingAuraFlow().patch_aura(model, shift)[0]


# ── attention override ──────────────────────────────────────────────────
ATTN_TARGETS = {"pytorch": "attention_pytorch", "ck_int8": "attention_comfy_kitchen_int8"}


def with_attention(model, mode, counter):
    """Force one attention backend for every attention call that goes
    through comfy's wrap_attn (transformer_options override wins over the
    per-block comfy_attention.config a MOSS int8 file carries). `counter`
    counts calls so a model that bypasses the override is caught."""
    target = getattr(attn_mod, ATTN_TARGETS[mode])

    def override(func, *args, **kwargs):
        counter[0] += 1
        return target(*args, **kwargs)

    cf = getattr(target, "container_function", None)
    if cf is not None:
        def override_c(*args, **kwargs):
            counter[0] += 1
            return cf(*args, **kwargs)
        override.container_function = override_c
    m = model.clone()
    to = dict(m.model_options.get("transformer_options", {}))
    to["optimized_attention_override"] = override
    m.model_options["transformer_options"] = to
    return m


# ── latents / sampling ──────────────────────────────────────────────────
def empty_latent(model, channels, down, w, h):
    """The tensor the family's Empty*LatentImage node makes (EmptyLatentImage
    4ch/8, EmptySD3LatentImage 16ch/8, EmptyFlux2LatentImage 128ch/16), then
    the same fix-up KSampler applies (channel repeat, 3-D latents)."""
    lat = torch.zeros([1, channels, h // down, w // down])
    return comfy.sample.fix_empty_latent_channels(model, lat, None)


def sample(model, positive, negative, latent, seed, steps, cfg, sampler, scheduler, step_cb=None):
    noise = comfy.sample.prepare_noise(latent, seed, None)

    def cb(step, x0, x, total):
        if step_cb:
            step_cb(step)
    return comfy.sample.sample(model, noise, steps, cfg, sampler, scheduler, positive, negative, latent,
                               denoise=1.0, callback=cb, disable_pbar=True, seed=seed)


def _sig(t, n):
    s = t.float().reshape(-1)
    return s if s.numel() == n else s.expand(n)


def _raw(x, denoised, sig):
    """(x - denoised) / sigma — the model's eps/velocity-space output for
    EPS and CONST (flow) parameterizations alike."""
    return (x - denoised) / sig.view(-1, *([1] * (x.ndim - 1))).to(x.dtype)


class Recorder:
    """model_function_wrapper on the bf16 reference: keeps every call's
    input and raw output, keyed by (sigma, cond/uncond)."""

    def __init__(self):
        self.data = {}

    def __call__(self, apply_model, args):
        x, t, c, cu = args["input"], args["timestep"], args["c"], args["cond_or_uncond"]
        out = apply_model(x, t, **c)
        sig = _sig(t, x.shape[0])
        raw = _raw(x, out, sig)
        per = x.shape[0] // len(cu)
        for i, lab in enumerate(cu):
            sl = slice(i * per, (i + 1) * per)
            self.data[(round(float(sig[i * per]), 6), int(lab))] = (x[sl].float().cpu(), raw[sl].float().cpu())
        return out


class TeacherForced:
    """model_function_wrapper on a variant: every call runs on the bf16
    reference's input for that (sigma, cond/uncond) instead of the variant's
    own trajectory, and its raw output is compared with the reference's.
    This isolates per-step error from trajectory divergence."""

    def __init__(self, ref):
        self.ref = ref
        self.stats = []
        self.missing = 0

    def __call__(self, apply_model, args):
        x, t, c, cu = args["input"], args["timestep"], args["c"], args["cond_or_uncond"]
        sig = _sig(t, x.shape[0])
        per = x.shape[0] // len(cu)
        xs, refs = [], []
        for i, lab in enumerate(cu):
            key = (round(float(sig[i * per]), 6), int(lab))
            r = self.ref.get(key)
            if r is None:
                self.missing += 1
                xs.append(x[i * per:(i + 1) * per])
                refs.append(None)
            else:
                xs.append(r[0].to(x.device, x.dtype))
                refs.append((key, r[1]))
        x_tf = torch.cat(xs)
        out = apply_model(x_tf, t, **c)
        raw = _raw(x_tf, out, sig)
        for i, r in enumerate(refs):
            if r is None:
                continue
            key, rraw = r
            a = raw[i * per:(i + 1) * per].float().flatten().cpu().double()
            b = rraw.flatten().double()
            self.stats.append({"sigma": key[0], "label": key[1],
                               "cos": float(F.cosine_similarity(a, b, dim=0)),
                               "rel": float((a - b).norm() / b.norm().clamp_min(1e-12))})
        return out


def save_ref(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(data)
    torch.save({"keys": keys, "x": [data[k][0] for k in keys], "raw": [data[k][1] for k in keys]}, path)


def load_ref(path):
    d = torch.load(path)
    return {tuple(k): (x, r) for k, x, r in zip(d["keys"], d["x"], d["raw"])}


# ── GPU bookkeeping ─────────────────────────────────────────────────────
def gpu_snapshot():
    """Utilization + other processes' memory on our GPU, so a run that shared
    the card is visible in the results."""
    try:
        import pynvml
        pynvml.nvmlInit()
        idx = torch.cuda.current_device()
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if vis and vis.split(",")[0].isdigit():
            idx = int(vis.split(",")[0])
        h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        u = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {"util": u.gpu, "used_gb": round(mem.used / 1e9, 2), "total_gb": round(mem.total / 1e9, 2)}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


def loaded_ratio():
    out = []
    for lm in mm.current_loaded_models:
        try:
            out.append(lm.model_loaded_memory() / max(1, lm.model_memory()))
        except Exception:
            pass
    return min(out) if out else None


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)
