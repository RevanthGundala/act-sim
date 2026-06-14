FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends \
        ca-certificates \
        build-essential \
        clang \
        curl \
        ffmpeg \
        git \
        linux-libc-dev \
        openssh-server \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        rsync \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv \
    && python -m pip install --no-cache-dir --upgrade pip uv \
    && python -m pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu124

WORKDIR /tmp/act-sim-deps
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY models ./models
COPY utils ./utils
RUN python -m pip install --no-cache-dir -e .

WORKDIR /workspace
CMD ["bash", "-lc", "mkdir -p /workspace && tail -f /dev/null"]
