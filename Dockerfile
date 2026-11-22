# syntax=docker/dockerfile:1
# Multi-stage zero-dependency Rust build for cachepin (linux/amd64 + linux/arm64)

FROM rust:1.86-bookworm AS builder
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN RUSTFLAGS="-D warnings" cargo build --release --bin cachepin

FROM debian:bookworm-slim AS runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates openssl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin cachepin

COPY --from=builder /src/target/release/cachepin /usr/local/bin/cachepin
RUN chmod 0755 /usr/local/bin/cachepin

LABEL org.opencontainers.image.title="cachepin"
LABEL org.opencontainers.image.description="Zero-dep Rust+SIMD lossless multi-agent LLM prompt-cache compiler, exact DP solver, auditor & Shadow-KV router"
LABEL org.opencontainers.image.source="https://github.com/pgg898200-ship-it/cachepin"
LABEL org.opencontainers.image.url="https://pgg898200-ship-it.github.io/cachepin/"
LABEL org.opencontainers.image.licenses="MIT OR Apache-2.0"
LABEL io.modelcontextprotocol.server.name="io.github.pgg898200-ship-it/cachepin"

USER cachepin
EXPOSE 47820
ENTRYPOINT ["/usr/local/bin/cachepin"]
CMD ["mcp"]
