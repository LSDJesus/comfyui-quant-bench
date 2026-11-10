"""What the quant benchmark runs: families, models, sampling, prompts, variants.

Everything here is data; prep.py / bake_all.py / run.py / report.py read it.
Paths are host paths (the bench containers mount /mnt/cold, /mnt/flash and
/mnt/working at the same paths on both hosts).

Sampling settings follow ComfyUI's own workflow templates for each family
(comfyui_workflow_templates 0.11.100): TE / VAE / latent node / sampler /
scheduler / steps / cfg. Krea 2 Raw has no template; Comfy's docs give 52
steps at cfg 3.5, we run 30 at 3.5 (the common community setting) to keep
the 3090 pass within a working day. A finetune samples like the model it
was tuned from.
"""

import os

# QBENCH_MNT prefixes every /mnt/... path, for a host that keeps the tree
# somewhere else (native Windows: QBENCH_MNT=D:/qbench-win).
MNT = os.environ.get("QBENCH_MNT", "")
ROOT = os.environ.get("QBENCH_ROOT", f"{MNT}/mnt/flash/downloads/_quant_bench")
COLD = f"{MNT}/mnt/cold/models"
FLASH = f"{MNT}/mnt/flash/models"
LORA_ROOT = f"{FLASH}/loras"

WIDTH = HEIGHT = 1024
SEEDS = (424242,)            # one seed per prompt; same seed for every variant

# ── variants ─────────────────────────────────────────────────────────────
# Main formats, in chart order. "bake" names the baker that produces the file
# (None = the source itself); "load_dtype" is ComfyUI's UNETLoader weight
# dtype (the plain cast); "emulate" disables that format's native compute
# (comfy.ops.get_disabled_quant_formats) so it runs dequant-to-bf16 —
# weight-only — the way it runs on GPUs without the kernels.
VARIANTS = {
    "bf16":     {"bake": None,    "label": "bf16 / fp16 (source)"},
    "fp8_cast": {"bake": None,    "label": "fp8 cast (UNETLoader fp8_e4m3fn)", "load_dtype": "fp8_e4m3fn"},
    "fp8":      {"bake": "fp8",   "label": "fp8 scaled"},
    "int8":     {"bake": "int8",  "label": "int8 ConvRot"},
    "w4a8":     {"bake": "w4a8",  "label": "w4a8 (ConvRot int4 + codebook)"},
    "nvfp4":    {"bake": "nvfp4", "label": "nvfp4"},
}
# Diagnostics (PRO 6000 only, PyTorch attention only): split each 4-bit /
# fp8 format's loss into its pieces.
DIAG_VARIANTS = {
    "fp8_wo":    {"bake": "fp8",       "label": "fp8 scaled, weight-only", "emulate": "float8_e4m3fn"},
    "nvfp4_wo":  {"bake": "nvfp4",     "label": "nvfp4, weight-only (bf16 activations)", "emulate": "nvfp4"},
    "w4a8_nocb": {"bake": "w4a8_nocb", "label": "w4a8, uniform levels (no codebook)"},
}
MAIN_ORDER = list(VARIANTS)
ATTENTIONS = ("pytorch", "ck_int8")    # PyTorch SDPA, Comfy Kitchen int8 attention

BAKERS = {   # bake name -> (script, extra args)
    "fp8":       ("bake_fp8.py", []),
    "int8":      ("bake_int8convrot.py", []),
    "w4a8":      ("bake_w4a8.py", []),
    "w4a8_nocb": ("bake_w4a8.py", ["--no-codebook"]),
    "nvfp4":     ("bake_nvfp4.py", []),
}

# ── prompts (same three subjects per family, phrased for its text encoder) ──
_NL = [
    "Close-up portrait of an elderly fisherman with a deeply weathered face, a grey knit cap and a salt-and-pepper beard, "
    "soft window light from the left, fine skin texture and individual hairs visible, shallow depth of field.",
    "A busy night market street in the rain: glowing neon shop signs, reflections on wet cobblestones, people with "
    "umbrellas, steam rising from food stalls, strings of lanterns overhead, cinematic wide shot.",
    "A ceramic coffee mug on a wooden desk beside an open notebook and a fountain pen; the mug is printed with the words "
    "\"GOOD MORNING\" in bold letters; warm morning sunlight, crisp product photography.",
]
_ANIMA = [
    "masterpiece, best quality, score_7, safe. An elderly fisherman with a weathered face, grey knit cap and beard, "
    "close-up portrait, soft window light, detailed.",
    "masterpiece, best quality, score_7, safe. A busy night market street in the rain, neon signs, reflections on wet "
    "stones, people with umbrellas, steam from food stalls, lanterns, wide shot.",
    "masterpiece, best quality, score_7, safe. A coffee mug on a wooden desk beside an open notebook and a fountain pen, "
    "the mug reads \"GOOD MORNING\", warm morning sunlight, still life.",
]
_SDXL = [
    "masterpiece, best quality, 1boy, old man, fisherman, wrinkled skin, grey knit cap, beard, portrait, close-up, "
    "window light, detailed face, realistic",
    "masterpiece, best quality, scenery, night, market, street, rain, neon lights, wet ground, reflections, crowd, "
    "umbrella, food stall, steam, lantern, wide shot",
    "masterpiece, best quality, still life, no humans, coffee mug, text \"GOOD MORNING\", wooden desk, open notebook, "
    "fountain pen, morning, sunlight",
]

NEG = {
    "anima": "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia",
    "zimage": "",
    "moss": "nsfw, worst quality, bad anatomy, watermark, jpeg artifact",
    "krea2": "",
    "sdxl": "lowres, worst quality, low quality, bad anatomy, bad hands, jpeg artifacts, watermark, signature, nsfw",
}

# ── families ─────────────────────────────────────────────────────────────
# te: (path, CLIPLoader type)   latent: (channels, downscale)
FAMILIES = {
    "anima": {
        "te": (f"{COLD}/text_encoders/anima/qwen_3_06b_base.safetensors", "stable_diffusion"),
        "vae": f"{FLASH}/vae/anima/qwen_image_vae.safetensors",
        "latent": (4, 8), "prompts": _ANIMA,
    },
    "zimage": {
        "te": (f"{COLD}/text_encoders/qwen_3_4b.safetensors", "lumina2"),
        "vae": f"{FLASH}/vae/Flux/ae.safetensors",
        "latent": (16, 8), "aura_shift": 3.0, "prompts": _NL,
    },
    "moss": {
        # comfy's template uses the w4a8 GPTQ TE; int8 ConvRot is the closest
        # to full precision on disk. One TE for every variant either way.
        "te": (f"{FLASH}/text_encoders/moss/qwen3.5_9b_int8_convrot.safetensors", "moss_image"),
        "vae": f"{FLASH}/vae/Flux.2_D/flux2-vae1.safetensors",
        "latent": (128, 16), "prompts": _NL,
    },
    "krea2": {
        # The official qwen3vl_4b is not on disk; this is the comfy-packed
        # abliterated Qwen3-VL-4B (fp8). Shared by every Krea 2 variant.
        "te": (f"{FLASH}/text_encoders/qwen3-4b-vl/qwen3VL4BAbliteratedComfyui_v10_full_fp8.safetensors", "krea2"),
        "vae": f"{FLASH}/vae/qwen/qwen_image_vae.safetensors",
        "latent": (4, 8), "prompts": _NL,
    },
    "sdxl": {
        "te": None,          # each checkpoint's own CLIP-L/G (see MODELS[...]["ckpt"])
        "vae": None,         # and its own VAE
        "latent": (4, 8), "prompts": _SDXL,
    },
}

_BASE = {"anima": dict(sampler="euler", scheduler="simple", steps=30, cfg=4.0),
         "zimage": dict(sampler="res_multistep", scheduler="simple", steps=25, cfg=4.0),
         "moss": dict(sampler="er_sde", scheduler="simple", steps=25, cfg=2.5),
         "krea2": dict(sampler="euler", scheduler="simple", steps=30, cfg=3.5),
         "sdxl": dict(sampler="dpmpp_2m", scheduler="karras", steps=25, cfg=5.5)}
_TURBO = {"anima": dict(sampler="euler", scheduler="simple", steps=8, cfg=1.0),
          "zimage": dict(sampler="res_multistep", scheduler="simple", steps=8, cfg=1.0),
          "moss": dict(sampler="euler", scheduler="simple", steps=8, cfg=1.0),
          "krea2": dict(sampler="euler", scheduler="simple", steps=8, cfg=1.0),
          "sdxl": dict(sampler="lcm", scheduler="sgm_uniform", steps=4, cfg=1.0)}

SRC = f"{ROOT}/sources"      # extracted SDXL UNets + DMD2 merges land here

DMD2 = {"moss": f"{LORA_ROOT}/moss/8-steps-dmd2-lora.safetensors",
        "sdxl": f"{LORA_ROOT}/illustrious/turbo/dmd2_sdxl_4step_lora_fp16.safetensors"}

# id -> model. "src" = the diffusion-model file every variant is baked from.
# "kind": base | turbo | turbo_lora | finetune. A turbo_lora row bakes nothing
# of its own: it runs the BASE model's files with the DMD2 LoRA applied at
# load (ComfyUI's LoraLoaderModelOnly), at full precision.
MODELS = {
    # Anima (2B, Cosmos-Predict2 derived)
    "anima_base":  {"family": "anima", "kind": "base", "name": "Anima base v1.0",
                    "src": f"{FLASH}/diffusion_models/anima/anima-base-v1.0.safetensors", **_BASE["anima"]},
    "anima_turbo": {"family": "anima", "kind": "turbo", "name": "Anima turbo v1.0",
                    "src": f"{FLASH}/diffusion_models/anima/anima-turbo-v1.0.safetensors", **_TURBO["anima"]},
    "anima_ft":    {"family": "anima", "kind": "finetune", "name": "CyberRealistic Anima v6.0",
                    "src": f"{COLD}/diffusion_models/anima/cyberrealisticAnima_v60AestheticV11.safetensors", **_BASE["anima"]},
    # Z-Image (6B NextDiT)
    "zimage_base":  {"family": "zimage", "kind": "base", "name": "Z-Image Base",
                     "src": f"{COLD}/diffusion_models/ZiB/zImageBase_base.safetensors", **_BASE["zimage"]},
    "zimage_turbo": {"family": "zimage", "kind": "turbo", "name": "Z-Image Turbo",
                     "src": f"{COLD}/diffusion_models/ZiT/zImageTurbo_turbo.safetensors", **_TURBO["zimage"]},
    "zimage_ft":    {"family": "zimage", "kind": "finetune", "name": "CyberRealistic ZiT v7.0",
                     "src": f"{COLD}/diffusion_models/ZiT/CyberRealistic_zit_v7.0_bf16.safetensors", **_TURBO["zimage"]},
    # MOSS-Image
    "moss_base":  {"family": "moss", "kind": "base", "name": "MOSS-Image base",
                   "src": f"{FLASH}/diffusion_models/moss/fp16/moss_image_base_bf16.safetensors", **_BASE["moss"],
                   "official": {"int8": f"{FLASH}/diffusion_models/moss/int8/moss_image_base_int8_convrot.safetensors"}},
    "moss_turbo": {"family": "moss", "kind": "turbo", "name": "MOSS-Image base + DMD2 8-step (merged)",
                   "src": f"{SRC}/moss_base_dmd2_merged.safetensors",
                   "merge": ("moss_base", DMD2["moss"], 1.0), **_TURBO["moss"]},
    "moss_turbo_lora": {"family": "moss", "kind": "turbo_lora", "name": "MOSS-Image base + DMD2 8-step (runtime LoRA)",
                        "uses": "moss_base", "lora": (DMD2["moss"], 1.0), **_TURBO["moss"]},
    "moss_ft":    {"family": "moss", "kind": "finetune", "name": "jibMix MOSS Realistic v1",
                   "src": f"{FLASH}/diffusion_models/moss/fp16/jibMixMossRealistic_v1OneToOne_bf16.safetensors", **_BASE["moss"]},
    # Krea 2 (12B)
    "krea2_base":  {"family": "krea2", "kind": "base", "name": "Krea 2 Raw",
                    "src": f"{FLASH}/diffusion_models/krea2/krea2_raw_bf16.safetensors", **_BASE["krea2"]},
    "krea2_turbo": {"family": "krea2", "kind": "turbo", "name": "Krea 2 Turbo",
                    "src": f"{FLASH}/diffusion_models/krea2/krea2_turbo_bf16.safetensors", **_TURBO["krea2"]},
    "krea2_ft":    {"family": "krea2", "kind": "finetune", "name": "CyberRealistic Krea 2 v4.0",
                    "src": f"{COLD}/diffusion_models/krea2/bf16/krea2Cyberrealistic_v40_bf16.safetensors", **_TURBO["krea2"]},
    # SDXL (Illustrious). "ckpt" = the full checkpoint (CLIP + VAE come from it);
    # prep.py extracts its UNet to "src".
    # The turbo rows put DMD2 on the FINETUNE, not the base: on the raw
    # Illustrious-XL v2.0 base every DMD2 LoRA tried (stock and the
    # Illustrious-tuned ones) renders washed-out grey, while the stock
    # 4-step LoRA on Luminelle looks right at its own LCM/4-step settings
    # (bf16 probe, 2026-11-08). Illustrious turbo merges are built this way.
    "sdxl_base":  {"family": "sdxl", "kind": "base", "name": "Illustrious-XL v2.0",
                   "ckpt": f"{FLASH}/checkpoints/Illustrious/Illustrious-XL-v2.0.safetensors",
                   "src": f"{SRC}/sdxl_base_unet.safetensors", **_BASE["sdxl"]},
    "sdxl_turbo": {"family": "sdxl", "kind": "turbo", "name": "Luminelle v1.0 + DMD2 4-step (merged)",
                   "ckpt": f"{FLASH}/checkpoints/Illustrious/Luminelle_v10.safetensors",
                   "src": f"{SRC}/sdxl_ft_dmd2_merged_unet.safetensors",
                   "merge": ("sdxl_ft", DMD2["sdxl"], 1.0), **_TURBO["sdxl"]},
    "sdxl_turbo_lora": {"family": "sdxl", "kind": "turbo_lora", "name": "Luminelle v1.0 + DMD2 4-step (runtime LoRA)",
                        "ckpt": f"{FLASH}/checkpoints/Illustrious/Luminelle_v10.safetensors",
                        "uses": "sdxl_ft", "lora": (DMD2["sdxl"], 1.0), **_TURBO["sdxl"]},
    "sdxl_ft":    {"family": "sdxl", "kind": "finetune", "name": "Luminelle v1.0 (Illustrious)",
                   "ckpt": f"{FLASH}/checkpoints/Illustrious/Luminelle_v10.safetensors",
                   "src": f"{SRC}/sdxl_ft_unet.safetensors", **_BASE["sdxl"]},
}

#: The order runs go in: the big four-format story first, the rest after.
FAMILY_ORDER = ("zimage", "anima", "moss", "krea2", "sdxl")

# ── fit pass (fit.py, every card) ─────────────────────────────────────────
#: The base and the turbo of each family, smallest family first. A finetune
#: has its base's shape, so it fits and runs at the same speed.
FIT_MODELS = ["sdxl_base", "sdxl_turbo", "anima_base", "anima_turbo", "zimage_base", "zimage_turbo",
              "moss_base", "moss_turbo", "krea2_base", "krea2_turbo"]
#: bf16 runs as the speed baseline when its file is under this ("fits a 16 GB
#: card": every family but Krea 2) or fits the card it runs on.
FIT_BF16_MAX_BYTES = 16e9


def fit_variants(model_id: str, vram_bytes: float, blackwell: bool, fp8: bool = False, bf16_bytes: float = None) -> list:
    """Formats the fit pass runs for one model on one card. fp8 = the card runs
    fp8 natively but not fp4 (RTX 40, Ada), so fp8 scaled is its own 8-bit float
    format. bf16_bytes stands in for the file size when the file isn't local
    (the community kit plans its downloads before it has them)."""
    vs = []
    bf = file_for(model_id, "bf16")
    size = bf16_bytes if bf16_bytes is not None else (os.path.getsize(bf) if os.path.exists(bf) else None)
    if size is not None and (size < FIT_BF16_MAX_BYTES or size < 0.9 * vram_bytes):
        vs.append("bf16")
    if fp8:
        vs.append("fp8")
    vs += ["int8", "w4a8"]
    if blackwell:
        vs.append("nvfp4")
    return vs


def bake_path(model_id: str, bake: str) -> str:
    return f"{ROOT}/bakes/{model_id}/{bake}.safetensors"


def file_for(model_id: str, variant: str) -> str:
    """The diffusion-model file a (model, variant) run loads."""
    m = MODELS[model_id]
    src_id = m.get("uses", model_id)          # turbo_lora runs the base's files
    v = {**VARIANTS, **DIAG_VARIANTS}[variant]
    if v["bake"] is None:
        return MODELS[src_id]["src"]
    return bake_path(src_id, v["bake"])


def cond_path(model_id: str) -> str:
    m = MODELS[model_id]
    key = m["family"] if m["family"] != "sdxl" else os.path.basename(m["ckpt"]).rsplit(".", 1)[0]
    return f"{ROOT}/cond/{key}.pt"


def ref_path(model_id: str, prompt_idx: int) -> str:
    return f"{ROOT}/ref/{model_id}/p{prompt_idx}.pt"


def run_dir(gpu: str, model_id: str, variant: str, attention: str) -> str:
    return f"{ROOT}/runs/{gpu}/{model_id}/{variant}__{attention}"
