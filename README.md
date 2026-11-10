# ComfyUI quantization bench: community kit

Run the fit-and-speed pass from the [ComfyUI Quantization Bench](https://ai-evolution.dev/quant-bench/)
on your own NVIDIA card and send the results back. Your card then shows up on
the page next to the ten already there, credited to you.

**Most wanted: RTX 40-series (Ada) cards.** They're the one generation in no
test yet, and the only one with fp8 hardware but no fp4, so they run a format
(fp8 scaled) nobody else's numbers cover. Other cards are welcome too.

## What it does

For each model family (Anima, Z-Image, MOSS-Image, Krea 2, SDXL/Illustrious),
base and turbo:

- loads every format your card can use: **int8 ConvRot** and **w4a8** on every
  card, **fp8 scaled** on RTX 40, **nvfp4** on RTX 50, and the **bf16**
  original when it's under 16 GB or fits your card
- times one 1024×1024 image after a warm-up, records how much of the model
  stayed on the card, and runs the same per-step accuracy check as the report

It runs stock ComfyUI in Docker, pinned to the exact versions behind the
published numbers (ComfyUI `bb878b9ee`, torch 2.12.0 + CUDA 13.0), so your
results are comparable with everyone else's.

## What you need

- An NVIDIA GPU, with **driver 580 or newer** (CUDA 13)
- **Linux** with Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  or **Windows 11** with Docker Desktop (WSL2 backend) and an Ubuntu WSL shell
- Disk space for the models: about **230 GB** for everything on a 24 GB card,
  less on smaller cards, and much less with `--families` (see below)
- 32 GB of system RAM or more is best; models that don't fit the card stream from it
- A couple of hours with the PC otherwise idle

## Run it

```bash
git clone https://github.com/LSDJesus/comfyui-quant-bench
cd comfyui-quant-bench
./run.sh
```

On Windows, run those three lines in your **WSL (Ubuntu) terminal**, inside
your WSL home folder (`cd ~` first), not on `/mnt/c`. If `docker` isn't found
there, turn on Docker Desktop → Settings → Resources → **WSL integration** for
your Ubuntu distro, then open a new terminal.

`run.sh` checks your setup, asks which GPU to test and how to credit you,
builds the container, shows exactly what it will download and how big it is,
asks before downloading, runs the bench, and packages the results. It can be
stopped and started again at any point; finished downloads and runs are kept.

Then open the [results form](https://github.com/LSDJesus/comfyui-quant-bench/issues/new?template=results.yml)
and drag `results/qbench-<your card>.zip` into it. Afterwards, `./run.sh --clean`
deletes the downloaded models.

### Less to download

```bash
./run.sh --families zimage,sdxl      # pick from: anima, zimage, moss, krea2, sdxl
```

Download per family on a 24 GB RTX 40 card: Anima 22 GB, SDXL 27 GB,
MOSS 49 GB, Z-Image 58 GB, Krea 2 72 GB (all five: 228 GB). Partial results
are still useful.

## What gets sent

The zip holds only what the bench recorded: one `result.json` per run (timings,
how much of the model fit, clocks and power mid-run, the accuracy check), the
rendered test image at 512 px, and `card.json`: GPU name, VRAM, driver, PCIe
link, power limit, system RAM, plus the credit name you typed, your OS, and
CPU model. Nothing else from your machine.

## Good to know

- **Use the PC as little as possible while it runs.** A card that drives your
  monitors shares time with the desktop; it costs a few percent, more on the
  fast models.
- **Windows:** WSL2 gets half your RAM by default. If you have 32 GB or less,
  raise it in `%UserProfile%\.wslconfig` (`[wsl2]` then `memory=28GB`, then
  `wsl --shutdown`).
- **Power limits and overclocks:** stock settings give the most comparable
  numbers. If yours are tuned, say so in the form.

## Files

The model files are on Hugging Face at
[LSDJesus/comfyui-quant-bench-files](https://huggingface.co/LSDJesus/comfyui-quant-bench-files):
the bf16 originals, the quantized bakes the report tested, and the reference
tensors for the accuracy check. `kit/manifest.json` lists every file with its
size and SHA-256 (`docker compose run --rm bench python /opt/kit/fetch.py --verify`
checks them). Each model keeps its original license; see the files repo.

The bench code in `bench/` is the same code that produced the report.
