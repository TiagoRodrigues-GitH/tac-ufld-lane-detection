# TAC-UFLD: reproducible training / evaluation / UI image.
#
#   docker build -t tac-ufld:gpu .                                   # CUDA 12.6 PyTorch (default)
#   docker build -t tac-ufld:cpu --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu .
#
# Datasets, results, checkpoints and the torch weight cache are NEVER copied
# into the image: mount them (see docker-compose.yml and docs/DOCKER.md).

FROM python:3.12-slim

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu126
ARG TORCH_VERSION=2.8.0
ARG EXTRAS=ui,deploy,dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/cache/torch \
    ELAS_ROOT=/data/elas \
    CULANE_ROOT=/data/culane \
    TUSIMPLE_ROOT=/data/tusimple \
    OPENLANE_ROOT=/data/openlane \
    TAC_UFLD_RESULTS=/app/results

# libgl/glib: OpenCV runtime; ffmpeg: video decoding for streaming; git: `package`.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) PyTorch from the index matching the target (CUDA or CPU), cached in its own layer.
RUN pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"

# 2) Package metadata first so dependency layers survive source edits.
COPY pyproject.toml README.md ./
COPY src/tac_ufld/__init__.py src/tac_ufld/__init__.py
RUN pip install -e ".[${EXTRAS}]" && pip uninstall -y tac-ufld

# 3) Source, configs, tests, docs (see .dockerignore for what is excluded).
COPY . .
RUN pip install --no-deps -e . && mkdir -p /app/results /data /cache/torch

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3).read()==b'ok' else 1)" || exit 1

# Default: environment check. Override the command for anything else, e.g.
#   docker run ... tac-ufld:gpu python -m tac_ufld run --config configs/elas_pilot.yaml
CMD ["python", "-m", "tac_ufld", "doctor"]
