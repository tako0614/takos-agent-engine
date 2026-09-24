# takos-agent-engine

takos-agent-engine は、Rust 製のサービスに LLM (大規模言語モデル) の agent を組み込むためのライブラリです。会話の履歴と
長期記憶を同じ記憶基盤で扱えるため、context window (モデルが一度に読める入力量の上限) に収まらなくなった情報も捨てずに
構造化し、覚え続ける agent を作れます。

これは単体で動くサービスではなく、ライブラリです。engine 本体は状態を持たず、storage / LLM / embedding / tool
実行はすべて trait 経由で注入します。engine library の開発はこの repository で行い、HTTP server などで engine を包む
service wrapper は ecosystem checkout の `takos/containers/agent/` が持ちます。agent runtime の境界は
[Agent Runtime](docs/agent-runtime.md) を参照してください。

## できること

- session 履歴と長期記憶を、RawNode (発話や tool の結果をそのまま残す一次記録) と AbstractNode
  (一次記録を構造化した知識) の二層で、同じ基盤の上に持てます。
- context window から溢れた情報を捨てず、蒸留 (raw の記録から entity と relation を取り出して構造化する処理)
  で知識として残せます。
- agent の実行フローを graph として定義し、checkpoint (実行途中の状態の保存点) から再開できます。
- LLM / embedding / storage / tool 実行を trait で注入するため、in-memory のテストからファイル永続化、分散ストレージまで
  backend を差し替えられます。
- graph の step 数・tool の往復数・timeout に上限があり、実行が止まらなくなるのを防ぎます。
- session 履歴、embedding、graph、provenance は session 境界で絞り、model が別 session を指定しても current
  session へ上書きします。
- tool の引数・結果、1 round の call 数、session / loop の読込件数、memory tool の hit 数を設定上限で制限します。
- side-effecting tool の timeout / cancellation を「失敗」と決めつけず、結果不明として停止し、同じ
  idempotency key で安全に fence できる実装だけを明示的な復旧対象にします。
- agent が自分の記憶を能動的に検索するための typed tool を 4 つ同梱しています。

## 使い方

### インストール

crate は `publish = false` で、crates.io には公開していません。利用側は git の revision を固定して依存します。
MSRV (Minimum Supported Rust Version) は Rust 1.85 です。

```toml
[dependencies]
takos-agent-engine = { git = "https://github.com/tako0614/takos-agent-engine", rev = "<commit-sha>", features = ["openai-embeddings"] }
```

利用できる feature flag:

- `openai-embeddings` — OpenAI 互換の embeddings backend を有効にします

`OpenAiEmbeddingConfig` と `OpenAiCompatibleEmbedder` の `Debug` 表示は API key を `[REDACTED]`
に置き換えます。これは secret を安全に保存する仕組みではないため、key の供給・rotation・process
環境の保護は組み込み側が担当します。

### example の実行

repo を clone して example を動かす場合:

```bash
cargo build
cargo run --example demo
cargo run --example object_demo
```

入口になる関数は `run_turn` / `run_turn_with_options` / `resume_loop` の 3 つです。まずは `examples/demo.rs` と
`examples/object_demo.rs` を読むのが早道です。

### 実行プロファイルの選択

`run_turn` の既定値は、後方互換の memory-aware graph です。発話の取り込みから蒸留までを engine が行うので、
記憶の管理を engine に任せる場合はそのまま使えます。

会話履歴・記憶・現在の turn のやり取りを組み込み先の製品側で管理し、engine にはその内容を渡すだけにする場合は、
`RunOptions.execution_profile = ExecutionProfile::ExternalContext` を明示し、model と tool の呼び出しだけを行う軽量な
graph を使います。

```rust
use takos_agent_engine::{ExecutionProfile, RunOptions};

let options = RunOptions {
    execution_profile: ExecutionProfile::ExternalContext,
    conversation_history: product_owned_history,
    ..RunOptions::default()
};
```

2 つのプロファイルの中身の違いは、後述の「実行プロファイル」節で説明します。

### 設定

設定は TOML で記述します。

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
max_graph_search_hits = 128
max_provenance_raw_nodes = 128
max_timeline_search_limit = 100
max_tool_argument_bytes = 262144
max_tool_result_bytes = 1048576

[runtime]
max_graph_steps = 64
max_tool_rounds = 8
node_timeout_ms = 10000
model_timeout_ms = 60000
tool_timeout_ms = 30000
distillation_timeout_ms = 15000
maintenance_batch_size = 32
max_tool_calls_per_round = 16
max_session_nodes = 512
max_loop_nodes = 256
```

`EngineConfig::validate` は reserve の合計、activation ratio の演算 overflow、NaN / 範囲外 threshold、0 timeout、
過大な collection / payload / step budget を起動前に拒否します。`RunOptions` の override も同じ上限を越えて runtime
budget を拡張できません。

### カスタム graph

標準の `run_turn` を使わず、独自の実行フローも組めます。

```rust
use std::sync::Arc;

let mut graph = ExecutionGraph::new("my_start_node");
graph.add_node(Arc::new(MyCustomNode));
graph.add_node(Arc::new(AnotherNode));
graph.add_edge("my_start_node", DEFAULT_EDGE, "another_node");

let runner = GraphRunner::new(Arc::new(graph));
let result = runner.run(state, config, deps, options).await?;
```

`GraphNode` trait を実装すれば、任意の処理を node として組み込めます。`NodeOutcome::Branch("edge_name")`
で条件分岐もできます。

## 仕組み

### 全体像

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

### 実行プロファイル

- `ExecutionProfile::MemoryAware` (既定) — user / tool / assistant の発話を RawNode として取り込み、session の読み込み、
  記憶の活性化、context の組み立て、session から押し出された記録の印付け (overflow marking)、蒸留までを行う 14 node の
  graph です。既存の利用側に向けた後方互換プロファイルです。
- `ExecutionProfile::ExternalContext` — 履歴と記憶の選択を製品側が持つ場合の 5 node の軽量 graph です。engine 側での
  取り込み・session の再読込・embedding と活性化・tool 結果の永続化・overflow 処理・蒸留は行わず、system prompt、
  `conversation_history`、現在の `user_message`、tool-call ID で対応づけた `turn_messages` だけを、それぞれ一度だけ
  model に渡します。

どちらのプロファイルも、`GraphRunner` の graph step の上限、tool 往復の上限、node / model / tool の timeout、
cancellation、checkpoint、native な tool-call ID の対応づけ、`SessionResponse.turn_messages` を共有します。
プロファイルは checkpoint にも記録され、異なるプロファイルでは再開できません。

### 二層記憶モデル

#### RawNode — 一次記録

生の発話、tool の結果、event をそのまま保存します。session 履歴でもあり、長期記憶の生ログでもあります。session
と記憶を別のものとしては保存しません。

- 5 種類: `UserUtterance` / `AssistantUtterance` / `ToolResult` / `Note` / `Event`
- vector 検索の対象
- `distillation_state` による蒸留ライフサイクルの追跡
- `was_pushed_out_of_session` による overflow 状態の管理
- `importance` (0.0-1.0) による活性化時の重み付け
- `operation_key` による first-writer-wins の canonical 書き込みと重複防止

#### AbstractNode — 構造化知識

RawNode の集まりから蒸留した知識の単位です。entity と relation からなる graph の断片を持ち、由来 (provenance)
として元の raw を参照できます。

- title + summary のテキスト表現
- graph の断片: entity と重み付き relation (主語-述語-目的語)
- raw / abstract への参照 (逆リンク)
- `abstraction_level` / `confidence` / `importance` のメタデータ
- vector 検索と graph 探索の両方の対象

### 設計の考え方

#### 溢れた記憶を捨てない

一般的な LLM agent は、context window を超えた情報を捨てます。takos-agent-engine は、session から押し出された発話や
tool の結果を「消える情報」ではなく「まだ構造化されていない原石」として扱います。overflow した raw node
は緩めた閾値で再活性化しやすくなり、maintenance pass で構造化された AbstractNode
に昇格します。情報は捨てられるのではなく、形を変えて生き続けます。

#### 圧縮ではなく構造化

蒸留は、token を減らすための「要約」ではありません。raw な出来事の集まりから entity と relation を取り出し、由来付きの
knowledge graph の断片として保存する処理です。AbstractNode は「何が起きたか」だけでなく「どの raw
から導かれたか」を参照でき、agent は自分の記憶の根拠を遡れます。

#### 状態を持たない core

engine 本体は状態を一切持ちません。session の状態、記憶、embedding の index、graph、checkpoint
はすべて注入された backend に委ねます。engine は「状態を持つ agent システムを動かす、状態を持たない実行部」であり、
backend を差し替えるだけで in-memory のテスト、ファイルへの永続化、分散ストレージのいずれにも対応できます。

#### 実行も記憶も graph

agent の実行フローは ExecutionGraph として宣言的に定義し、GraphRunner が 1 node ずつ進めます。記憶も AbstractNode の
relation graph として蓄積します。実行と記憶の両方が graph なので、agent
の振る舞いと知識の構造を同じ考え方で扱えます。

### 実行フロー

memory-aware の標準 agent は、14 node の上限付き multi-step graph として実装しています。

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

tool の往復は 1 回固定ではありません。`run_model` と `execute_tools` は条件付きで複数回往復できますが、
`max_tool_rounds` を超えません。

### 記憶の活性化

活性化 (activation) は、保存済みの記憶から今の turn に関係するものを選び出す処理です。活性化の query
は以下から組み立てます。

- 現在の user message
- plan (ある場合)
- 直近の session context
- 直前の tool の結果 (再活性化時)

score は複数のシグナルを合成して決めます。

| シグナル            | 説明                                                 |
| ------------------- | ---------------------------------------------------- |
| semantic similarity | embedding の cosine 類似度                           |
| importance bias     | node の importance 値による加算                      |
| time decay          | 経過日数に応じた減衰 (0.015/日)                      |
| overflow bonus      | session から押し出された未蒸留 node への加算 (+0.12) |

Raw と Abstract は別々に検索し、`target_ratio` (既定 1:1) に従って配分します。overflow した raw は閾値を緩め (0.72 →
0.63)、再活性化しやすくします。

### context の組み立て

context window は token の予算 (budget) として管理します。`ContextAssembler` が次のように配分します。

```
total_tokens (64K)
├─ reserve_system (4K)
├─ reserve_tools (12K)
├─ reserve_working (8K)
└─ remaining → session (50%) + memory (50%)
```

session 分は時系列に入るだけ詰め、入りきらなかった raw は `pushed_out` として overflow marking
の対象になります。memory 分は Abstract を優先し、次に Raw を詰めます。

### 記憶を探索する tool

記憶は自動の活性化だけでなく、agent が能動的に探索できます。そのための typed tool を 4 つ提供します。

| ツール                   | 説明                                                                |
| ------------------------ | ------------------------------------------------------------------- |
| `semantic_search_memory` | current session の raw/abstract を embedding で検索                  |
| `graph_search_memory`    | current session の relation graph を深さ・hit 数制限付きで探索       |
| `provenance_lookup`      | current session の AbstractNode から上限付きで raw 群を逆引き        |
| `timeline_search`        | current session の raw を期間・件数制限付きで時系列検索              |

標準 graph は 4 tool の引数を実行前に parse し、現在の `session_id` を強制して model
指定値を信用しません。depth / hit / provenance / timeline / top-k は config 上限へ clamp
し、vector hit を node に戻す段階でも session を再確認します。`NodeRepository` などの低レベル API
には管理・移行用途の unscoped query がありますが、model-visible memory tool の権限境界には使いません。

### 蒸留

蒸留は、現在の loop の RawNode 群を入力として AbstractNode を生成する処理です。

- session / loop を表す entity の作成
- raw node との relation の付与
- tool の結果との relation の付与
- 活性化されていた abstract への `informed_by` relation の付与
- relation への由来 raw id の記録

蒸留に成功した raw は `Distilled` に更新し、overflow による加点を外します。session
から溢れたまま未蒸留の backlog は `run_maintenance_pass` で後から処理します。current-loop distillation と
maintenance が同じ `(session_id, loop_id)` を同時に処理しないよう、repository の lease / fence を取得してから
実行します。claim は入力 raw ID の digest と固有 lease ID を持ち、期限切れまたは所有権を失った worker
は結果を commit できません。abstract の operation key は入力 digest から決まり、crash 後の再実行でも重複生成しません。

### checkpoint と再開

GraphRunner は、副作用の境界の前後で LoopState を保存します。process の再起動や明示的な pause で
`LoopStatus::Paused` の checkpoint が残っている場合は、`resume_loop` で直前の node から再開できます。

```rust
// 中断した loop を再開
resume_loop(config, deps, session_id, loop_id, options).await?;
```

LoopState は ExecutionState 全体を JSON として直列化したもので、活性化の snapshot、session window
の判断、実行待ちの tool call を含みます。`resume_loop` が再開の対象にするのは `LoopStatus::Paused` の checkpoint
だけです。cancellation / timeout は `LoopStatus::Cancelled` / `LoopStatus::TimedOut` として checkpoint
に残り、状態は失われませんが、そのまま自動では再開しません。checkpoint には schema version、実行 profile 固有の
`graph_id`、外側と内側の `session_id` / `loop_id` を持たせ、どれかが一致しない状態は安全側に停止して拒否します。
永続 recovery を使う caller は `RunOptions.loop_id` に安定した ID を渡します。省略時の自動生成 ID は一時的な
run 向けです。

process が `Running` checkpoint を残した場合は `recover_interrupted_loop_with_options` を明示的に使います。model
request は「何度実行しても結果が同じ」を provider-neutral に保証する契約がないため再送しません。tool node も `ToolExecutor::recovery_is_idempotent`
が true の call だけを、元と同じ idempotency key で再実行します。

checkpoint / resume は、永続的な `LoopStateRepository` を注入する利用側に向けたライブラリの部品です。Takos の service
wrapper は、container 内に置いた checkpoint を障害復旧の復元元にはせず、run のたびに Takos Worker
が正とする履歴から状態を組み立て直します。

### trait による注入

LLM / embedding / 蒸留など、特定のベンダーに依存する実装は crate 本体に持たず、trait 越しに注入します。
永続化については `storage` module が公開の object backend (`FileObjectStore`,
`ObjectNodeRepository`, `ObjectVectorIndex`, `ObjectGraphRepository`, `ObjectLoopStateRepository`) を export します。

| Trait                 | 責務                                                                                     |
| --------------------- | ---------------------------------------------------------------------------------------- |
| `ModelRunner`         | LLM の呼び出し。会話と tool のやり取り + context → 出力 (message + 対応づけた tool call) |
| `Embedder`            | テキスト → embedding vector                                                              |
| `ToolExecutor`        | tool call → 対応づけた結果 + 読み取り専用 / 副作用ありの実行ポリシー                     |
| `Distiller`           | raw node 群 → AbstractNode + ライフサイクル更新                                          |
| `ScoringPolicy`       | similarity / importance / decay / overflow → 最終 score                                  |
| `TokenEstimator`      | テキスト → token 数の推定                                                                |
| `NodeRepository`      | raw / abstract の CRUD + timeline / session / loop の query                              |
| `VectorIndex`         | embedding の index + 類似検索                                                            |
| `GraphRepository`     | relation graph の index + 深さ制限付き探索                                               |
| `LoopStateRepository` | checkpoint の save / load / clear                                                        |

`ToolExecutor::execute_with_context` には `session_id`、`loop_id`、安定した `idempotency_key`、timeout、
`CancellationToken` を渡します。side-effecting executor はこの key を実際の永続 boundary まで伝播し、子 process /
request の停止も担当します。read-only call の timeout は通常の tool error として model に返せますが、dispatch
後の side-effecting call が timeout / cancellation になった場合は、remote side effect の有無を判断できないため
`EngineError::ToolOutcomeIndeterminate` で run を停止します。engine はそれを成功・失敗へ推測せず、自動 retry
もしません。

demo 用の実装は `examples/common/support.rs` と test 用の support に閉じています。例外は `openai-embeddings` feature
を有効にした場合だけで、このとき OpenAI 互換の embeddings backend を公開します。

### storage backend

永続 baseline はファイルベースの object backend です。JSON object を正とする情報として保存し、query
を速くするための index を別に生成します。公開 in-memory backend は deterministic test / prototype
用で、process を越える recovery authority にはしません。

```
store.json                              # format version, metadata
.store.lock                             # process 間の advisory lock
raw/{id}.json                           # RawNode
abstract/{id}.json                      # AbstractNode
embeddings/raw/{id}.json                # raw embedding
embeddings/abstract/{id}.json           # abstract embedding
graph/{id}.json                         # relation graph
checkpoints/{session}--{loop}.json      # LoopState
journal/current.json                    # 未完了 mutation の write-ahead journal
receipts/
  raw_operation/{digest}.json           # operation_key → canonical raw id
  abstract_operation/{digest}.json      # operation_key → canonical abstract id
claims/distillation/{session}--{loop}.json # distillation lease / fence
quarantine/                             # 壊れた通常 object の隔離先
indexes/
  session/{session_id}.json             # session → raw id list
  loop/{loop_id}.json                   # loop → raw id list
  timeline/raw/{YYYYMMDD}.json           # 日別 timeline shard
  backlog/undistilled_raw.json          # maintenance 対象
  vector/raw/{session_id}.json           # session 別 embedding manifest
  vector/abstract/{session_id}.json
```

raw commit は node・operation receipt・timeline/session/loop/backlog index・embedding body / manifest を、
abstract commit は node・receipt・embedding・graph projection を 1 つの write-ahead journal mutation
として扱います。raw lifecycle の複数 node 更新も journal に記録します。process が途中で落ちた場合、次の open
が journal を、何度 replay しても結果が同じ形で適用してから index の完全性を検査し、必要なら canonical
object から再構築します。
receipt は index rebuild と別の永続 identity であり、digest と元の `operation_key` の両方を照合します。

同じ root の更新は process 内 mutex と `.store.lock` の OS advisory lock の両方で直列化します。root と管理対象 path
の symlink は拒否し、root 外へ抜ける path を書きません。通常 object の壊れた JSON は `quarantine/` に隔離して
Storage error を返しますが、未完了 mutation の唯一の記録である壊れた journal は自動削除・隔離せず、operator
介入まで安全側に停止します。hard kill で残った古い staging `*.tmp` は open 時に回収します。

この file backend の journal と lock は単一 filesystem root の整合性境界です。network filesystem や分散 writer
へそのまま拡張できる分散 transaction / lease ではありません。その場合は各 trait の永続実装側で同等以上の
atomicity と fencing を提供します。

### 実行の上限

実行が止まらなくなるのを防ぐため、実行には上限 (budget) を設けています。

| パラメータ                | デフォルト | 説明                                           |
| ------------------------- | ---------- | ---------------------------------------------- |
| `max_graph_steps`         | 64         | graph node の最大実行数                        |
| `max_tool_rounds`         | 8          | tool loop の最大往復数                         |
| `max_tool_calls_per_round` | 16         | 1 model round の tool call 上限                |
| `max_session_nodes`       | 512        | context 用に読む直近 session node 上限         |
| `max_loop_nodes`          | 256        | 1 回の distillation が扱う loop node 上限      |
| `node_timeout_ms`         | 10,000     | 通常 node のタイムアウト                       |
| `model_timeout_ms`        | 60,000     | model node のタイムアウト                      |
| `tool_timeout_ms`         | 30,000     | tool 1 call ごとのタイムアウト                 |
| `distillation_timeout_ms` | 15,000     | 蒸留のタイムアウト                             |
| `maintenance_batch_size`  | 32         | maintenance pass の取得件数                    |

`RunOptions` では、run ごとの上書き、`CancellationToken`、製品側が管理する `conversation_history`、`ExecutionProfile`
を渡せます。0 や config より大きい override で上限を無効化することはできません。model が返した tool call
数・ID / name / arguments は side effect の dispatch 前に検証し、上限超過は call を黙って捨てず run
を安全側に停止します。

tool result は `max_tool_result_bytes` 以内の envelope に縮約してから engine storage へ保存します。さらに model
に見せる current-turn transcript 全体を `context_budget.reserve_tools` 内へ収めます。大きい内容は
`{"truncated":true,"preview":"..."}` の形になりますが、tool-call ID と tool-result message 自体は保持します。
完全な出力が必要な tool は、実行側の永続 artifact / object に先に保存し、engine には bounded な参照を返してください。

## 開発

```bash
cargo fmt --all --check
cargo check --locked --lib --no-default-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
cargo +1.85.0 check --locked --all-targets --all-features
cargo run --example demo
cargo run --example object_demo
```

CI は stable で fmt / default library check / clippy / test / rustdoc を、Rust 1.85 で全 target・全 feature の check
を実行します。dependency 解決の漂流を避けるため、すべての compile/test gate は committed `Cargo.lock`
を `--locked` で使います。

## 関連リンク

- [`docs/agent-runtime.md`](docs/agent-runtime.md) — agent runtime の境界
- [`examples/demo.rs`](examples/demo.rs) / [`examples/object_demo.rs`](examples/object_demo.rs) — 最小の組み込み例
- `takos/containers/agent/` — Takos の service wrapper (ecosystem checkout 内)

## ライセンス

MIT License で公開しています。
