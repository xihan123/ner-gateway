# syntax=docker/dockerfile:1
# Build arguments
ARG ONNX_RUNTIME_VERSION=1.24.2

# Stage 1: Build
FROM rust:1.93.1-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies with cache mount
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev

# Copy Cargo files first for dependency caching
COPY backend/Cargo.toml backend/Cargo.lock ./

# Create dummy main.rs to build dependencies first (better caching)
RUN --mount=type=cache,target=/app/target \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    mkdir -p src && \
    echo "fn main() {}" > src/main.rs && \
    cargo build --release && \
    rm -rf src

# Copy actual source and build
COPY backend/src ./src

# Build with cached dependencies
RUN --mount=type=cache,target=/app/target \
    --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/usr/local/cargo/git \
    touch src/main.rs && \
    cargo build --release && \
    cp target/release/ner-gateway /usr/local/bin/ner-gateway

# Stage 2: Runtime
FROM debian:bookworm-slim

# Re-declare ARG for this stage
ARG ONNX_RUNTIME_VERSION

# Install runtime dependencies and ONNX Runtime (combined for fewer layers)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    --mount=type=cache,target=/tmp/onnx-cache \
    apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libssl3 \
    ca-certificates \
    curl \
    wget \
    && cache_file="/tmp/onnx-cache/onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}.tgz" \
    && if [ ! -f "$cache_file" ]; then \
        wget -q -O "$cache_file" \
          "https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_RUNTIME_VERSION}/onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}.tgz"; \
    fi \
    && tar -xzf "$cache_file" -C /tmp \
    && mv "/tmp/onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}/lib/libonnxruntime"*.so* /usr/lib/ \
    && rm -rf "/tmp/onnxruntime-linux-x64-${ONNX_RUNTIME_VERSION}" \
    && apt-get remove -y wget \
    && apt-get autoremove -y \
    && ldconfig

# Copy the binary
COPY --from=builder /usr/local/bin/ner-gateway /usr/local/bin/ner-gateway

# Create working directory
WORKDIR /app

# Create data and models directories
RUN mkdir -p /app/data /app/models

# Copy frontend
COPY backend/index.html /app/index.html

# Create non-root user
RUN useradd -m -r -s /bin/false appuser && \
    chown -R appuser:appuser /app

# Set environment variables
ENV NER_MODEL_PATH=/app/models/ner_model_int8.onnx
ENV NER_VOCAB_PATH=/app/models/vocab.txt
ENV NER_DB_PATH=/app/data/ner_reviews.db
ENV NER_PORT=8080
ENV ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so

# Expose port
EXPOSE 8080

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start service
ENTRYPOINT ["ner-gateway"]