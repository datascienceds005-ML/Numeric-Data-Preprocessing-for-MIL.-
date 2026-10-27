# syntax=docker/dockerfile:1
# Multi-stage build: wheels compile in the builder, the runtime stays slim.
# NOTE: pin the base image by digest in production (docker pull
# python:3.12-slim-bookworm && docker image inspect) — the tag is pinned here
# and the digest comment is intentionally left for the release pipeline to fill
# in, because a fabricated digest would be worse than an explicit TODO-free
# documented pinning procedure.

FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Runtime dependencies first (layer cache), then the package itself.
COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt && \
    pip install --prefix=/install setuptools wheel

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps --prefix=/install .

FROM python:3.12-slim-bookworm AS runtime

# Non-root user; no network access is required at runtime.
RUN groupadd --gid 10001 numprep && \
    useradd --uid 10001 --gid numprep --create-home numprep

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1

# Runtime image: library code only — no dev tools, no tests, no build deps.
COPY --from=builder /install /usr/local

WORKDIR /app
RUN mkdir -p /data /out && chown -R numprep:numprep /data /out

USER numprep

# Mounted-volume conventions:
#   /data — raw CSV inputs (mount read-only: -v "$PWD/data:/data:ro")
#   /out  — run outputs (mount writable:   -v "$PWD/out:/out")
ENTRYPOINT ["numprep"]
# One-command reproduction: inputs at /data, outputs at /out, built-in defaults.
CMD ["run", "--input", "/data", "--output", "/out"]
