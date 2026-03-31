# takos-agent-engine

Takos の agent engine を Rust で実装する standalone repository です。

現在は library-first の単一 crate 構成で、公開面は 2 層です。

- `run_turn` / `run_turn_with_options` / `resume_loop` / `run_maintenance_pass`: 標準の Takos agent preset
- `ExecutionGraph` / `GraphRunner`: node 分割された graph runtime

session と memory は同じ storage substrate 上で扱い、状態は library 内ではなく injected backend に保持します。`RawNode` と `AbstractNode` の二層 memory、context budget による session/memory 配分、overflow-aware retrieval、checkpoint/resume、bounded multi-step tool loop、object-backed 永続化と idempotent persistence を備えています。SQLite backend は optional adapter として残しています。

## Commands

```bash
cargo build
cargo test
cargo run --example demo
cargo run --example object_demo
cargo run --example sqlite_demo
```
