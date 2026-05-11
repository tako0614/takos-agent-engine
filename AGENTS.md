# AGENTS.md — takos-agent-engine

`takos-agent-engine` は Takos の **stateless agent runtime library** (Rust) で、 session 履歴と長期 memory を同一基盤で
扱い、 RawNode / AbstractNode の二層記憶を activation しながら checkpoint 可能な graph runtime で長期継続実行する。
agent runtime の正本 library であり、 service wrapper は `takos/agent/` が持つ。

## 責務

### 持つ

- GraphRunner (checkpointable graph execution)
- RawNode / AbstractNode 二層 memory
- token-budget context assembly
- embedding vector search
- activation scoring
- distillation (raw → AbstractNode 昇格)
- checkpoint / resume
- LLM provider trait / memory backend trait の inject 点
- maintenance pass (overflow / structure 変換)

### 持たない

- service wrapping (HTTP server、 RPC binding 等は `takos/agent/` の責務)
- production vendor implementation (OpenAI / Claude 等は feature gate のみ提供)
- Takos-specific orchestration (skill catalog、 system prompt 等は `takos/agent/` 側)

## 隣接 product との contract

- **Upstream**: なし (library)
- **Downstream consumer**: `takos/agent/` (Takos の agent execution service)

## Substitutability

library 内部は **代替可能**: LLM provider / memory backend / vector store を trait 経由で inject する設計。 production
vendor は feature gate で生かし、 unit test は mock-llm / mock-vector で回す。

## Workflow

```bash
cd takos-agent-engine
cargo build
cargo test
cargo test --features mock-llm
cargo fmt --check
cargo clippy
```

## 関連 docs

- [`README.md`](README.md) — engine の設計理念と memory model
- [`docs/agent-runtime.md`](docs/agent-runtime.md) — agent runtime の境界
