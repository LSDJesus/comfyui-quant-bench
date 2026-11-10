# The software the published results ran on: Ubuntu 24.04, Python 3.12,
# torch 2.12.0 + CUDA 13.0, ComfyUI bb878b9ee (v0.43.0 + 27 commits), and the
# package versions from that environment (requirements.txt). Keeping it
# identical is what makes your card's numbers comparable with everyone else's.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv git ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r /tmp/requirements.txt

ARG COMFY_COMMIT=bb878b9ee35274e67931634ba9e8c338f8c06035
RUN git clone --filter=blob:none https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI \
    && git -C /opt/ComfyUI checkout -q ${COMFY_COMMIT}

COPY bench /opt/bench
COPY kit /opt/kit
ENV COMFY_DIR=/opt/ComfyUI QBENCH_DYNAMIC=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    HOME=/tmp/qhome HF_HUB_DISABLE_TELEMETRY=1
WORKDIR /opt/bench
