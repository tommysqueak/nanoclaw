# syntax=docker/dockerfile:1.7
# NanoClaw Agent Container
# Runs Claude Agent SDK in isolated Linux VM with browser automation.
#
# Runtime split:
#   - agent-runner (our TypeScript code): Bun, mounted RO at /app/src by host
#   - globally-installed Node CLIs (claude-code, agent-browser, vercel): pnpm + Node
#
# Source is never baked in — /app/src is provided by a shared read-only
# bind mount at runtime (see src/container-runner.ts). Source-only changes
# never require an image rebuild.

# ---- Stage 1: build whisper.cpp ---------------------------------------------
# whisper.cpp is built from source so we get the actual whisper-cli binary
# (the npm package "whisper-cli" is unrelated — it wraps OpenAI's API and
# defeats sovereignty). RPATH=$ORIGIN/../lib lets the binary find its sibling
# libs without LD_LIBRARY_PATH gymnastics in the runtime stage.
FROM debian:bookworm-slim AS whisper-builder
ARG WHISPER_CPP_VERSION=v1.8.3
ARG WHISPER_MODEL_FILE=ggml-base.en.bin
RUN apt-get update && apt-get install -y --no-install-recommends \
        cmake \
        build-essential \
        libssl-dev \
        pkg-config \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${WHISPER_CPP_VERSION} \
        https://github.com/ggerganov/whisper.cpp.git /src/whisper.cpp
RUN cd /src/whisper.cpp && \
    cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX=/whisper-install \
        -DCMAKE_INSTALL_RPATH='$ORIGIN/../lib' \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE && \
    cmake --build build -j --config Release && \
    cmake --install build
RUN mkdir -p /whisper-install/share && \
    curl -fsSL --retry 5 --retry-delay 2 --retry-max-time 300 -C - \
        -o /whisper-install/share/${WHISPER_MODEL_FILE} \
        https://huggingface.co/ggerganov/whisper.cpp/resolve/main/${WHISPER_MODEL_FILE}

# ---- Stage 2: parakeet model -------------------------------------------------
# NVIDIA Parakeet TDT 0.6B v3 (multilingual) ONNX model for onnx-asr.
# Int8 quantized for CPU. ~670MB total. Only downloaded when INSTALL_PARAKEET=true.
# Also downloads Silero VAD model (~2MB) for long-form audio splitting.
FROM debian:bookworm-slim AS parakeet-builder
ARG INSTALL_PARAKEET=true
ARG PARAKEET_MODEL_REPO=istupakov/parakeet-tdt-0.6b-v3-onnx
ARG SILERO_VAD_REPO=istupakov/silero-vad-onnx
RUN if [ "$INSTALL_PARAKEET" = "true" ]; then \
        apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
        && rm -rf /var/lib/apt/lists/* \
        && mkdir -p /parakeet /parakeet-vad \
        && for f in config.json vocab.txt encoder-model.int8.onnx decoder_joint-model.int8.onnx; do \
             curl -fsSL --retry 5 --retry-delay 2 --retry-max-time 300 -C - \
               -o /parakeet/$f \
               "https://huggingface.co/${PARAKEET_MODEL_REPO}/resolve/main/$f"; \
           done \
        && curl -fsSL --retry 5 --retry-delay 2 --retry-max-time 60 \
             -o /parakeet-vad/silero_vad.onnx \
             "https://huggingface.co/${SILERO_VAD_REPO}/resolve/main/silero_vad.onnx"; \
    else \
        mkdir -p /parakeet /parakeet-vad; \
    fi

# ---- Stage 3: runtime --------------------------------------------------------
FROM node:22-slim

# ---- Build-time arguments ----------------------------------------------------
# CJK fonts add ~200MB. Opt in only if you render Chinese/Japanese/Korean text.
ARG INSTALL_CJK_FONTS=false
# Parakeet v3 ONNX model adds ~700MB (model + Python + onnx-asr). Opt out with
# INSTALL_PARAKEET=false to skip.
ARG INSTALL_PARAKEET=true

# Pin CLI versions for reproducibility. Bump deliberately — unpinned installs
# mean every rebuild silently picks up the latest and can break in lockstep
# across all users.
ARG CLAUDE_CODE_VERSION=2.1.128
ARG AGENT_BROWSER_VERSION=latest
ARG VERCEL_VERSION=52.2.1
ARG BUN_VERSION=1.3.12
ARG WS_VERSION=8.20.0
ARG WHISPER_MODEL_FILE=ggml-base.en.bin
ARG GMAIL_MCP_VERSION=1.1.11
ARG CALENDAR_MCP_VERSION=2.6.1

# ---- System dependencies -----------------------------------------------------
# tini: correct PID 1 / signal forwarding so outbound.db writes finalize on
# SIGTERM instead of being orphaned by the shell entrypoint.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        chromium \
        fonts-liberation \
        fonts-noto-color-emoji \
        libgbm1 \
        libnss3 \
        libatk-bridge2.0-0 \
        libgtk-3-0 \
        libx11-xcb1 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libasound2 \
        libpangocairo-1.0-0 \
        libcups2 \
        libdrm2 \
        libxshmfence1 \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        tini \
        unzip \
    && if [ "$INSTALL_CJK_FONTS" = "true" ]; then \
        apt-get install -y --no-install-recommends fonts-noto-cjk; \
       fi \
    && rm -rf /var/lib/apt/lists/*

# Chromium path for agent-browser / Playwright consumers
ENV AGENT_BROWSER_EXECUTABLE_PATH=/usr/bin/chromium
ENV PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium
# Belt-and-braces: prevent Playwright's postinstall from downloading its own
# ~300MB Chromium. We've already installed the system one above.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

# ---- Bun runtime -------------------------------------------------------------
# Install via the official script (handles multi-arch detection), then move
# the binary to /usr/local/bin so the non-root `node` user can execute it.
RUN curl -fsSL https://bun.sh/install | bash -s "bun-v${BUN_VERSION}" && \
    install -m 0755 /root/.bun/bin/bun /usr/local/bin/bun && \
    rm -rf /root/.bun

# ---- agent-runner deps -------------------------------------------------------
# Deps are cached independently of CLI versions. Source is NOT baked in —
# it's provided by the shared RO mount at runtime.
WORKDIR /app

COPY agent-runner/package.json agent-runner/bun.lock ./

RUN --mount=type=cache,target=/root/.bun/install/cache \
    bun install --frozen-lockfile

# ---- pnpm + global Node CLIs -------------------------------------------------
# Most stable first, most frequently bumped last. Bumping claude-code
# (the most common change) only invalidates one layer.
#
# only-built-dependencies gates pnpm's supply-chain policy:
#   - agent-browser has a postinstall build step.
#   - @anthropic-ai/claude-code's postinstall downloads the native Claude
#     binary (linux-arm64 variant on our image). Without the allowlist
#     the SDK fails at spawn time with "native binary not found".
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
# Pin pnpm to match the host (package.json packageManager). pnpm 11 stopped
# honoring `only-built-dependencies[]=` in .npmrc for global installs, which
# silently skips claude-code's native-binary postinstall and agent-browser's
# bin chmod — the agent then crashes at runtime with "native binary not
# installed". Keep this in lockstep with package.json's `packageManager`.
ARG PNPM_VERSION=10.33.0
RUN corepack enable && corepack prepare pnpm@${PNPM_VERSION} --activate

RUN --mount=type=cache,target=/root/.cache/pnpm \
    echo "only-built-dependencies[]=agent-browser" > /root/.npmrc && \
    echo "only-built-dependencies[]=@anthropic-ai/claude-code" >> /root/.npmrc && \
    pnpm install -g "vercel@${VERCEL_VERSION}"

RUN --mount=type=cache,target=/root/.cache/pnpm \
    pnpm install -g "agent-browser@${AGENT_BROWSER_VERSION}"

# ws: install globally then symlink to /node_modules so ESM `import 'ws'` resolves
# from any working directory. ESM walks the directory tree upward; only /node_modules
# at the root is universally reachable regardless of cwd.
RUN npm install -g "ws@${WS_VERSION}" && \
    mkdir -p /node_modules && \
    ln -s /usr/local/lib/node_modules/ws /node_modules/ws

RUN --mount=type=cache,target=/root/.cache/pnpm \
    pnpm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}"

RUN --mount=type=cache,target=/root/.cache/pnpm \
    pnpm install -g \
        "@gongrzhe/server-gmail-autoauth-mcp@${GMAIL_MCP_VERSION}" \
        "@cocal/google-calendar-mcp@${CALENDAR_MCP_VERSION}" \
        "zod-to-json-schema@3.22.5"

# ---- ncl CLI wrapper ----------------------------------------------------------
# Actual script lives in the mounted source at /app/src/cli/ncl.ts.
RUN printf '#!/bin/sh\nexec bun /app/src/cli/ncl.ts "$@"\n' > /usr/local/bin/ncl && \
    chmod +x /usr/local/bin/ncl

# ---- whisper.cpp from builder stage -----------------------------------------
# Container-side voice transcription, sovereign by default. Audio attachments
# get transcribed inside the container via local Whisper.
COPY --from=whisper-builder /whisper-install/bin/whisper-cli /usr/local/bin/whisper-cli
COPY --from=whisper-builder /whisper-install/lib/ /usr/local/lib/
COPY --from=whisper-builder /whisper-install/share/${WHISPER_MODEL_FILE} /whisper/model.bin
RUN ldconfig
ENV WHISPER_MODEL_PATH=/whisper/model.bin

# ---- Parakeet v3 ONNX from builder stage ------------------------------------
# Python + onnx-asr for CPU inference. Conditional on INSTALL_PARAKEET build arg.
# Silero VAD model included for long-form audio (>25s) splitting.
COPY --from=parakeet-builder /parakeet /parakeet
COPY --from=parakeet-builder /parakeet-vad /parakeet-vad
RUN if [ -f /parakeet/config.json ]; then \
        apt-get update && apt-get install -y --no-install-recommends python3 python3-pip \
        && rm -rf /var/lib/apt/lists/* \
        && pip3 install --break-system-packages 'onnx-asr[cpu]' 'huggingface_hub' \
        && rm -rf /root/.cache/pip; \
    fi
ENV PARAKEET_MODEL_PATH=/parakeet
ENV PARAKEET_VAD_MODEL_PATH=/parakeet-vad
# ---- Entrypoint --------------------------------------------------------------
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# ---- Workspace + permissions -------------------------------------------------
RUN mkdir -p /workspace/group /workspace/extra && \
    chown -R node:node /workspace && \
    chmod 777 /home/node

USER node
WORKDIR /workspace/group

# tini is PID 1, reaps zombies, forwards signals cleanly. entrypoint.sh does
# `exec bun ...` so bun runs as tini's direct child.
ENTRYPOINT ["/usr/bin/tini", "--", "/app/entrypoint.sh"]
