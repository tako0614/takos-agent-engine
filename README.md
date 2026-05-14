# takos-agent-engine

Rust で書かれた Takos の agent engine library。 RawNode / AbstractNode の二層記憶を activation しながら、 checkpoint
可能な graph runtime で agent loop を回す。 engine 本体は stateless で、 storage / LLM / embedding / tool 実行はすべて
trait 経由で注入する。 LLM agent を embed したい Rust service の中で、 session 履歴と長期 memory を同一基盤で扱いたい
ときに使う。

この repository が engine library の正本であり、 service wrapper は ecosystem checkout の `takos/agent/` が持つ。 agent
runtime の境界は [Agent Runtime](docs/agent-runtime.md) を参照。

## Install / Quickstart

crate は `publish = false` で crates.io には公開していない。 consumer は git revision を pin して依存する。

```toml
[dependencies]
takos-agent-engine = { git = "https://github.com/tako0614/takos-agent-engine", rev = "<commit-sha>", features = ["openai-embeddings"] }
```

利用可能な feature flag:

- `openai-chat` — OpenAI-compatible chat backend を有効化
- `openai-embeddings` — OpenAI-compatible embeddings backend を有効化

repo を clone して example を動かす場合:

```bash
cargo build
cargo run --example demo
cargo run --example object_demo
```

`run_turn` / `run_turn_with_options` / `resume_loop` が high-level facade。 まずは `examples/demo.rs` と
`examples/object_demo.rs` を読むのが早い。

## アーキテクチャ概要

```
                    ┌─────────────────────────┐
                    │      run_turn()          │  High-level facade
                    │  run_turn_with_options() │
                    │  resume_loop()           │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │     GraphRunner          │  Graph execution engine
                    │  ExecutionGraph          │
                    │  GraphNode trait          │
                    └────────┬────────────────┘
                             │
     ┌───────────┬───────────┼───────────┬──────────────┐
     ▼           ▼           ▼           ▼              ▼
┌─────────┐ ┌─────────┐ ┌────────┐ ┌─────────┐ ┌───────────┐
│ Memory  │ │ Context │ │ Model  │ │  Tool   │ │  Storage  │
│Activation│ │Assembly │ │Runner  │ │Executor │ │  Layer    │
└─────────┘ └─────────┘ └────────┘ └─────────┘ └───────────┘
```

## 二層記憶モデル

### RawNode — 一次記録

生の発話、tool result、event をそのまま保存する。session 履歴でもあり、長期 memory の生ログでもある。session と memory
を別物として保存しない。

- 5 種類: `UserUtterance` / `AssistantUtterance` / `ToolResult` / `Note` / `Event`
- vector search の対象
- `distillation_state` で蒸留ライフサイクルを追跡
- `was_pushed_out_of_session` で overflow 状態を管理
- `importance` (0.0-1.0) で活性化時の重み付け
- `operation_key` で idempotent persistence

### AbstractNode — 構造化知識

RawNode 群から蒸留された knowledge unit。entity / relation の graph fragment を持ち、provenance として元の raw
を参照できる。

- title + summary のテキスト表現
- graph fragment: entities + weighted relations (subject-predicate-object)
- raw / abstract への参照 (backlinks)
- `abstraction_level` / `confidence` / `importance` メタデータ
- vector search と graph traversal の両方の対象

## 設計理念

### memory は session の外に溢れても死なない

一般的な LLM agent は context window を超えた情報を捨てる。takos-agent-engine は session
から押し出された発話やツール結果を「消える情報」ではなく「まだ構造化されていない原石」として扱う。overflow した raw node
は relaxed threshold で再活性化しやすくなり、maintenance pass で構造化された AbstractNode
に昇格する。情報は捨てられるのではなく、形を変えて生き続ける。

### 圧縮ではなく構造化

distillation は token を減らすための「要約」ではない。raw な出来事群から entity と relation を抽出し、provenance 付きの
knowledge graph fragment として保存する操作である。AbstractNode は「何が起きたか」だけでなく「どの raw
から導かれたか」を参照でき、agent は自分の記憶の根拠を遡れる。

### stateless core, stateful world

engine 本体は一切の状態を内包しない。session state、memory、embedding index、graph、checkpoint はすべて injected
dependency に委ねる。したがって engine は「stateful agent system を動かす stateless runner」であり、backend
を差し替えるだけで in-memory テスト、ファイルベース永続化、分散ストレージのいずれにも対応できる。

### graph として実行し、graph として記憶する

agent の実行フローは ExecutionGraph として宣言的に定義し、GraphRunner が 1 node ずつ進める。記憶もまた AbstractNode の
relation graph として蓄積される。実行と記憶の両方が graph であることで、agent
の振る舞いと知識構造が同じ概念モデルで扱える。

### checkpoint で止まっても壊れない

GraphRunner は side effect 境界の前後で checkpoint を取る。process restart や明示的な pause
で `LoopStatus::Paused` の checkpoint が残っている場合は、`resume_loop` で直前の node から再開できる。timeout /
cancellation は状態を失わないために checkpoint へ残すが、現在の public resume API はそれらを自動再開しない。tool
実行の途中で落ちても operation_key による idempotent persistence が二重書き込みを防ぐ。

## 実行フロー

標準 agent は 14 node の bounded multi-step graph として実装されている。

```
ingest_user_input
  → load_session_view
  → build_activation_query
  → activate_memory
  → assemble_context
  → run_model ─────────────────────────────────┐
       │                                        │
       ├─ (no tool calls) → persist_output      │
       │                                        │
       └─ (tool calls) → execute_tools          │
                           → build_followup_query│
                           → reactivate_memory   │
                           → reassemble_context  │
                           → run_model_after_tools
                                │
                                ├─ (more tools, rounds < max) → execute_tools ...
                                └─ (done) → persist_output
                                              → mark_session_overflow
                                              → distill_current_loop
                                              → finish
```

tool round は 1 回固定ではなく bounded multi-step。`run_model` と `execute_tools`
は条件付きで複数回往復できるが、`max_tool_rounds` を超えない。

## Memory Activation

activation query は以下から構成する。

- 現在の user message
- plan (ある場合)
- recent session context
- 直前の tool result (再活性化時)

scoring は複数シグナルを合成する。

| シグナル            | 説明                                                 |
| ------------------- | ---------------------------------------------------- |
| semantic similarity | embedding の cosine 類似度                           |
| importance bias     | node の importance 値による加算                      |
| time decay          | 経過日数に応じた減衰 (0.015/日)                      |
| overflow bonus      | session から押し出された未蒸留 node への加算 (+0.12) |

Raw と Abstract は別々に検索し、`target_ratio` (デフォルト 1:1) に従って配分する。overflow した raw は閾値を緩和 (0.72 →
0.63) して再活性化しやすくする。

## Context Assembly

context window は token budget として管理する。`ContextAssembler` が以下を配分する。

```
total_tokens (64K)
├─ reserve_system (4K)
├─ reserve_tools (12K)
├─ reserve_working (8K)
└─ remaining → session (50%) + memory (50%)
```

session bucket は時系列で greedy に詰め、入りきらなかった raw は `pushed_out` として overflow marking
の対象になる。memory bucket は Abstract 優先、次に Raw の順で詰める。

## Memory Exploration Tools

memory は自動活性化だけでなく、agent が能動的に探索できる typed tool を 4 つ提供する。

| ツール                   | 説明                                                                |
| ------------------------ | ------------------------------------------------------------------- |
| `semantic_search_memory` | embedding で raw/abstract を検索。target / top_k / threshold 指定可 |
| `graph_search_memory`    | AbstractNode から relation graph を depth-limited traversal         |
| `provenance_lookup`      | AbstractNode の元になった raw 群を逆引き                            |
| `timeline_search`        | 時系列で raw を検索。session scope / global scope 対応              |

## Distillation

distillation は current loop の RawNode 群を入力として AbstractNode を生成する。

- session / loop entity を立てる
- raw node との relation を張る
- tool result relation を張る
- activated abstract に `informed_by` relation を張る
- provenance raw ids を relation に埋める

distillation 成功後は raw を `Distilled` に更新し、overflow bias を落とす。session から溢れた backlog は
`run_maintenance_pass` で後処理する。

## Checkpoint & Resume

GraphRunner は side effect 境界の前後で LoopState を保存する。

```rust
// 中断した loop を再開
resume_loop(config, deps, session_id, loop_id, options).await?;
```

LoopState は ExecutionState 全体を JSON serialize し、activation snapshot、session window decision、pending tool calls
を含む。`resume_loop` が再開対象にするのは `LoopStatus::Paused` の checkpoint。cancellation / timeout は
`LoopStatus::Cancelled` / `LoopStatus::TimedOut` として checkpoint に残り、状態は失われないが、そのまま自動再開はしない。

## Vendor-Neutral Traits

LLM / embedding / distillation など vendor 依存の実装は crate 本体に持たず、trait 越しに注入する。
永続化については `storage` module が public object backend (`FileObjectStore`,
`ObjectNodeRepository`, `ObjectVectorIndex`, `ObjectGraphRepository`, `ObjectLoopStateRepository`) を export する。

| Trait                 | 責務                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `ModelRunner`         | LLM 呼び出し。input (system/session/memory/tool context) → output (message + tool calls) |
| `Embedder`            | テキスト → embedding vector                                                              |
| `ToolExecutor`        | tool call → result (name + content + summary)                                            |
| `Distiller`           | raw nodes → AbstractNode + lifecycle updates                                             |
| `ScoringPolicy`       | similarity / importance / decay / overflow → final score                                 |
| `TokenEstimator`      | テキスト → token 数推定                                                                  |
| `NodeRepository`      | raw / abstract の CRUD + timeline / session / loop query                                 |
| `VectorIndex`         | embedding の index + similarity search                                                   |
| `GraphRepository`     | relation graph の index + depth-limited traversal                                        |
| `LoopStateRepository` | checkpoint の save / load / clear                                                        |

demo 用の実装は `examples/common/support.rs` と test support に閉じている。
例外として `openai-chat` / `openai-embeddings` feature を有効にした場合だけ
OpenAI-compatible backend を公開する。chat backend は `OpenAiChatConfig` の
optional tool catalog を `tools` / `tool_choice` として request に載せ、response
の `tool_calls` は `ModelOutput` に変換する。

## Storage Backend

現在の正式 backend は file-based object backend。JSON object を正本にし、materialized index で query を高速化する。

```
store.json                              # format version, metadata
raw/{id}.json                           # RawNode
abstract/{id}.json                      # AbstractNode
embeddings/raw/{id}.json                # raw embedding
embeddings/abstract/{id}.json           # abstract embedding
graph/{id}.json                         # relation graph
checkpoints/{session}--{loop}.json      # LoopState
indexes/
  session/{session_id}.json             # session → raw id list
  loop/{loop_id}.json                   # loop → raw id list
  timeline/raw.json                     # global raw timeline
  backlog/undistilled_raw.json          # maintenance 対象
  vector/raw_embeddings.json            # embedding manifest
  vector/abstract_embeddings.json
```

open 時に canonical object から index を再整列し、不整合を自動修復する。`operation_key` による idempotent lookup index
も持つ。

## Runtime Budget

無限 loop を防ぐため、実行は budget を持つ。

| パラメータ                | デフォルト | 説明                            |
| ------------------------- | ---------- | ------------------------------- |
| `max_graph_steps`         | 64         | graph node の最大実行数         |
| `max_tool_rounds`         | 8          | tool loop の最大往復数          |
| `node_timeout_ms`         | 10,000     | 通常 node のタイムアウト        |
| `tool_timeout_ms`         | 30,000     | tool 実行のタイムアウト         |
| `distillation_timeout_ms` | 15,000     | 蒸留のタイムアウト              |
| `maintenance_batch_size`  | 32         | maintenance pass のバッチサイズ |

`RunOptions` で per-run の override と `CancellationToken` を渡せる。

## 設定

TOML で記述する。

```toml
system_prompt = "You are the Rust-based Takos agent engine."

[memory.activation]
top_k_total = 20
use_time_decay = true
overflow_raw_threshold_relaxation = true

[memory.activation.target_ratio]
raw = 1
abstract = 1

[memory.retrieval.similarity_threshold]
raw = 0.72
abstract = 0.74

[memory.retrieval]
relaxed_threshold_for_pushed_raw = 0.63

[context_budget]
total_tokens = 64000
reserve_system = 4000
reserve_tools = 12000
reserve_working = 8000
session_ratio = 0.5
memory_ratio = 0.5

[tools]
memory_search = true
graph_search = true
provenance_lookup = true
timeline_search = true
max_memory_search_top_k = 32
max_graph_search_depth = 4
max_timeline_search_limit = 100

[runtime]
max_graph_steps = 64
max_tool_rounds = 8
node_timeout_ms = 10000
tool_timeout_ms = 30000
distillation_timeout_ms = 15000
maintenance_batch_size = 32
```

## カスタム Graph

標準の `run_turn` を使わず、独自の実行フローを組める。

```rust
use std::sync::Arc;

let mut graph = ExecutionGraph::new("my_start_node");
graph.add_node(Arc::new(MyCustomNode));
graph.add_node(Arc::new(AnotherNode));
graph.add_edge("my_start_node", DEFAULT_EDGE, "another_node");

let runner = GraphRunner::new(Arc::new(graph));
let result = runner.run(state, config, deps, options).await?;
```

`GraphNode` trait を実装すれば任意の処理を node として組み込める。`NodeOutcome::Branch("edge_name")` で条件分岐も可能。

## Commands

```bash
cargo build
cargo test
cargo clippy --all-targets -- -D warnings
cargo run --example demo
cargo run --example object_demo
```

## License

AGPL-3.0-only
