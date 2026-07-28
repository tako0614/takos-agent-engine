# takos-agent-engine architecture v0.3

> このページでわかること: Takos agent engine (Rust) のアーキテクチャと設計方針。

## 1. 目的

`takos-agent-engine` は、Takos の agent loop を Rust で実装する library-first な runtime である。

この engine の役割は次の 4 点に集約される。

- session 履歴と長期 memory を単一基盤で扱う
- RawNode / AbstractNode の二層記憶を activation して context を再構成する
- node graph と checkpoint により paused loop を再開可能に実行する
- raw な出来事を構造化して provenance 付き graph memory に育てる

この文書は概念仕様ではなく、現在の Rust 実装の正本アーキテクチャを説明する。

## 2. 基本方針

### 2.1 library-first

公開入口は binary ではなく `lib.rs` である。標準入口は次の 2 層に分かれる。

- 低レベル: `ExecutionGraph` / `GraphRunner`
- 高レベル: `run_turn` / `run_turn_with_options` / `resume_loop` / `run_maintenance_pass`

demo wiring は `examples/` に置き、crate 本体は stateless core として保つ。

### 2.2 core は stateless

engine 本体は session state や memory state を内包しない。状態はすべて injected dependency 側に置く。

- node / abstract / timeline は `NodeRepository`
- embedding search は `VectorIndex`
- abstract relation graph は `GraphRepository`
- loop checkpoint は `LoopStateRepository`

したがって正確には「stateful agent system を動かす stateless runner」である。

### 2.3 append-friendly と fenced persistence

RawNode は event log として追記し、AbstractNode は構造化済み知識として append する。再開時に二重書き込みを避けるため、side effect には `operation_key` を持たせる。

- user input persist は `loop:{loop_id}:user_input`
- tool result persist は `loop:{loop_id}:tool:{round}:{index}:{name}`
- assistant output persist は `loop:{loop_id}:assistant_output`
- distillation persist は session / loop / 入力 raw ID digest / 出力 index から導出する

object backend は `operation_key` の digest だけでなく元 key も含む durable receipt を first-writer-wins
で保存する。tool の remote side effect には同じ key を `ToolExecutionContext.idempotency_key`
として渡し、engine 内の dedup と外部実行境界の fence を同じ identity へ揃える。

## 3. 現在のシステム全体像

標準 agent は bounded multi-step graph として実装されている。

```text
ingest_user_input
  -> load_session_view
  -> build_activation_query
  -> activate_memory
  -> assemble_context
  -> run_model
    -> execute_tools
    -> build_followup_activation_query
    -> reactivate_memory
    -> reassemble_context
    -> run_model_after_tools
    -> (tool rounds remain ? execute_tools : persist_assistant_output)
  -> persist_assistant_output
  -> mark_session_overflow
  -> distill_current_loop
  -> finish
```

`run_turn` はこの default graph を実行する facade であり、custom graph を直接 `GraphRunner` で動かすこともできる。
graph は `build_default_execution_graph` が組み立てる単一の linear pipeline で、model が tool call を返したときだけ
`execute_tools` へ分岐して再 activation pass を回す。

## 4. ディレクトリ構成

```text
src/
  lib.rs
  config.rs
  error.rs
  ids.rs
  domain/
  engine/
    execution_graph.rs
    session_engine.rs
    context_assembler.rs
  memory/
  storage/
  model/
  tools/
examples/
  demo.rs
  object_demo.rs
  common/support.rs
```

`src/main.rs` は持たず、binary wiring は `examples/` に限定する。

## 5. ドメインモデル

### 5.1 RawNode

RawNode は生の発話、tool result、途中メモ、event を表す一次記録である。

- graph edge は持たない
- vector search の対象になる
- `distillation_state` と `overflow` を持つ
- idempotent persist 用に `operation_key` を持てる

RawNode は session の近傍履歴でもあり、長期 memory の生ログでもある。session と memory を別物として保存しない。

### 5.2 AbstractNode

AbstractNode は RawNode 群から蒸留された構造化 knowledge unit である。

- raw provenance を参照できる
- 他の abstract を参照できる
- graph fragment を持つ
- vector search と graph traversal の両方の対象になる
- `operation_key` により再開時の重複生成を防げる
- producing session の `session_id` を持ち、model-visible graph / provenance query は session を越えない

### 5.3 LoopState

loop の一時状態と recovery 情報は `LoopState` に checkpoint される。主要フィールドは次の通り。

- `current_node`
- `checkpoint_version`
- `graph_id`
- `session_id`
- `loop_id`
- `iteration`
- `tool_rounds_completed`
- `model_invocations`
- `status`
- `last_completed_node`
- `last_effect_key`
- `recent_events`
- `activated_raw`
- `activated_abstract`
- `session_window`
- `pushed_out_raw`
- `tool_result_ids`
- `assistant_message`
- `state_json`

`state_json` には `ExecutionState` 全体を保存する。復元時には schema version、profile 固有の
`graph_id`、envelope と `state_json` の session / loop identity を照合し、不一致を
`RecoveryUnsafe` として拒否する。durable caller は `RunOptions.loop_id` に安定した ID
を渡し、省略時の自動生成 ID は一時 run にだけ使う。現在の public `resume_loop`
が再開対象にするのは `LoopStatus::Paused` の checkpoint であり、cancelled /
timed-out checkpoint は状態保存用で自動再開対象ではない。

### 5.4 LoopStatus

loop status は次を持つ。

- `Running`
- `Paused`
- `Finished`
- `Cancelled`
- `TimedOut`
- `Failed`

## 6. storage 抽象

最低限の境界は次の 4 つである。

- `NodeRepository`
- `VectorIndex`
- `GraphRepository`
- `LoopStateRepository`

### 6.1 NodeRepository

`NodeRepository` は raw / abstract の正本であり、timeline・loop・session view と raw lifecycle patch を担当する。

重要 API:

- `insert_raw`
- `insert_abstract`
- `insert_raw_once`
- `insert_abstract_once`
- `commit_raw`
- `commit_abstract`
- `get_raw_by_operation_key`
- `get_abstract_by_operation_key`
- `recent_session_raw`
- `session_raw`
- `raw_for_loop`
- `timeline_raw`
- `update_raw_lifecycle`
- `undistilled_raw`

`commit_raw` / `commit_abstract` は durable backend が node body と embedding、
必要な receipt / index / graph projection を 1 つの mutation boundary で commit
するための API である。単純 backend の既定実装は insert と projection update
を分けられるため、`NodeCommit::projections_committed` を見て engine が不足分を補う。
distillation 用 repository は bounded loop read と lease / fencing API も持ち、
同じ loop を current turn と maintenance が二重処理しないようにする。

### 6.2 VectorIndex

`VectorIndex` は raw / abstract の embedding index を持つ。現在の production baseline
である `ObjectVectorIndex` は exact cosine search を維持する。

### 6.3 GraphRepository

`GraphRepository` は AbstractNode の relation graph を index し、predicate filter、session scope、depth、
hit 数上限付き traversal を提供する。

戻り値は node id だけではなく、`depth` と `via_predicate` を含む `GraphTraversalHit` である。

### 6.4 LoopStateRepository

checkpoint の保存・読込・削除を担当する。実在する graph node であることを確認した後、その node の前で `Running` checkpoint
を保存し、pause / timeout / cancellation / failure では該当 status の checkpoint
を残す。再開 API が受け付けるのは `Paused` checkpoint である。

## 7. backend 実装

durable baseline は object backend である。公開されている in-memory 実装は deterministic test / prototype
用であり、process を越える recovery・locking・fencing authority にはしない。

object backend は JSON object を正本にし、session・memory・embedding・graph・checkpoint を同じ root 配下に保存する。query path では directory scan に頼らず、session / loop / timeline / backlog / embedding manifest を materialized index として持つ。

- `store.json`
- `.store.lock`
- `raw/{id}.json`
- `abstract/{id}.json`
- `embeddings/raw/{id}.json`
- `embeddings/abstract/{id}.json`
- `graph/{id}.json`
- `checkpoints/{session_id}--{loop_id}.json`
- `journal/current.json`
- `receipts/raw_operation/{digest}.json`
- `receipts/abstract_operation/{digest}.json`
- `claims/distillation/{session_id}--{loop_id}.json`
- `quarantine/`
- `indexes/session/{session_id}.json`
- `indexes/loop/{loop_id}.json`
- `indexes/timeline/raw/{YYYYMMDD}.json`
- `indexes/backlog/undistilled_raw.json`
- `indexes/backlog/pushed_undistilled_raw.json`
- `indexes/vector/raw/{session_id}.json`
- `indexes/vector/abstract/{session_id}.json`

運用上重要な点:

- `operation_key` receipt は materialized index の rebuild と独立した durable identity である
- raw / abstract commit と raw lifecycle update は write-ahead journal を先に書き、open 時に未完了 mutation
  を冪等 replay する
- object backend は journal replay 後に index の version・parse・完全性を検査し、必要なら canonical object
  から再構築する
- `store.json` に `format_version` / `created_at` / `updated_at` / `last_index_rebuild_at` を持つ
- undistilled backlog と overflow backlog に index を持つ
- timeline は日別、embedding manifest は session 別 shard とし、bounded query が installation 全体を読まない
- vector retrieval は session filter と deterministic ordering を行う
- 同じ root の mutation は process 内 mutex と `.store.lock` の OS advisory lock で直列化する
- root・管理 path の symlink と root 外 path を拒否する
- 通常 object の壊れた JSON は quarantine して error を返すが、未完了 mutation の壊れた journal
  は失わず fail-closed にする
- open 時に hard kill で残った古い staging temp file を回収する

この atomicity / lock は単一 filesystem root の境界であり、network filesystem や複数 host
の distributed transaction を意味しない。別 backend は同じ trait contract を自身の transaction /
conditional write / lease で実装する。

## 8. 実行モデル

### 8.1 ExecutionGraph / GraphRunner

`ExecutionGraph` は node と branch edge を持つ実行 graph である。`GraphRunner` は graph を 1 node ずつ進める。

node は `GraphNode` trait を実装する。

```rust
#[async_trait]
pub trait GraphNode: Send + Sync {
    fn id(&self) -> &'static str;
    fn runtime_class(&self) -> NodeRuntimeClass;
    async fn run(
        &self,
        state: &mut ExecutionState,
        config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome>;
}
```

`NodeOutcome` は `Continue` / `Branch` / `Finish` / `Pause` を持つ。

### 8.2 checkpoint / recovery

`GraphRunner` は node 実行前に `Running` checkpoint を保存し、loop
が止まった理由を checkpoint status に残す。

- `Pause`: `LoopStatus::Paused` として保存し、`resume_loop` で再開できる
- `Cancellation`: `LoopStatus::Cancelled` として保存するが、public resume API
  は自動再開しない
- `Timeout` / step budget 超過: `LoopStatus::TimedOut` として保存するが、public
  resume API は自動再開しない
- node error: `LoopStatus::Failed` として保存して error を返す

`resume_loop(config, deps, session_id, loop_id, options)` は checkpoint から
`ExecutionState` を復元するが、`GraphRunner::resume` は `Paused` 以外の
checkpoint を `EngineError::LoopTerminated(status)` として拒否する。

process crash が残した `Running` checkpoint は通常 resume と分け、
`recover_interrupted_loop_with_options` で明示的に扱う。model node は request
が billable completion を既に作った可能性を否定できないため再送しない。tool node
も全 pending call が `ToolExecutor::recovery_is_idempotent` を満たす場合だけ、
元と同じ engine idempotency key で再実行する。

### 8.3 bounded runtime

無限 loop を防ぐため、実行は runtime budget を持つ。

- `max_graph_steps`
- `max_tool_rounds`
- `max_tool_calls_per_round`
- `max_session_nodes`
- `max_loop_nodes`
- `node_timeout_ms`
- `model_timeout_ms`
- `tool_timeout_ms`
- `distillation_timeout_ms`
- `maintenance_batch_size`

tool round は 1 回固定ではなく bounded multi-step である。`run_model` と `execute_tools`
は条件付きで複数回往復できるが、`max_tool_rounds` を超えない。model 出力の call 数、
arguments、tool result envelope、memory query の depth / hit / limit も side effect
の前に検証・制限する。上限を越えた call は黙って drop せず fail-closed にする。

### 8.4 cancellation-safe

`RunOptions` は `CancellationToken` と timeout override を受け取れる。override
で設定上限を拡張したり 0 timeout にしたりはできない。`GraphRunner` は node runtime class ごとに timeout を適用する。

- `Standard`
- `ToolExecution`
- `Distillation`

通常 node の timeout や pre-dispatch cancellation は `LoopStatus::TimedOut` /
`LoopStatus::Cancelled` として checkpoint に残す。tool dispatch 後は呼び出し結果を
推測しない。read-only call の timeout は model-visible tool error にできる一方、
side-effecting call の timeout / cancellation は `ToolOutcomeIndeterminate`
として run を止め、idempotency key と理由を caller に返す。これらは状態を失わないための
terminal checkpoint であり、`resume_loop` の通常再開対象ではない。

## 9. activation 設計

activation query は次の材料から作る。

- 現在の user message
- plan
- recent session
- 直前の tool result

embedding は `Embedder` trait で生成する。activation は Raw / Abstract を別検索し、config の target ratio に従って採用する。初期値は 1:1 である。

score には少なくとも次が入る。

- semantic similarity
- importance bias
- freshness / time decay
- overflow bonus

未蒸留かつ session から押し出された raw は閾値を緩めて再活性化しやすくする。

## 10. context assembly

context window は token budget として扱う。`ContextAssembler` は次を組み立てる。

- system prompt
- recent session bucket
- activated memory bucket
- tool bucket

同時に session window decision を返し、何を含めて何を押し出したかを `ExecutionState` に保存する。

押し出された raw で未蒸留のものは、後段の overflow marking で relaxed retrieval の対象になる。
session load は `max_session_nodes` 件に制限し、external conversation history は assistant
tool-call と対応する全 tool result を不可分 group として末尾から budget 内へ選ぶ。orphan /
incomplete exchange を model へ送らず、checkpoint に保存する transcript も同じ bounded
representation にする。

## 11. model と tool

### 11.1 ModelRunner

model は `ModelRunner` trait 越しに使う。crate 本体は default では concrete model runner を公開せず、example と test support に deterministic toy 実装を置く。consumer (`takos/containers/agent`) は自前の chat runner を持ち込み、本 crate は embeddings backend (`openai-embeddings` feature) のみを OpenAI-compatible backend として公開する。

`OpenAiEmbeddingConfig` / `OpenAiCompatibleEmbedder` の `Debug` は API key
を `[REDACTED]` と表示する。これは secret storage の代替ではなく、key lifecycle
と実行環境の保護は consumer が所有する。

### 11.2 ToolExecutor

tool 実行は `ToolExecutor` trait 越しに行う。標準 graph では model が tool call を返したときだけ `ExecuteToolsNode` が走る。

tool result は次の性質を持つ。

- RawNode として永続化される
- `max_tool_result_bytes` 以内へ縮約した envelope を保持する
- `operation_key` により resume 後も重複しない
- follow-up activation query と reassembly に反映される

`ToolExecutor::execute_with_context` は `session_id` / `loop_id` / 安定した
`idempotency_key` / timeout / cancellation token を実行側へ渡す。未分類 call
は side-effecting として扱い、隣接 read-only call だけを並列化する。side-effecting
executor は key を remote durable boundary まで伝播し、child process / request
の cancellation cleanup を所有する。

engine は full result の外部 artifact authority ではない。large output が必要な tool
は自身の durable object に保存し、bounded preview/reference を返す。

## 12. memory exploration tools

memory は自動活性化だけでなく、tool として能動探索できる。

提供する typed tool は次の通り。

- `semantic_search_memory`
- `graph_search_memory`
- `provenance_lookup`
- `timeline_search`

設計上の原則:

- raw/abstract の hit は deterministic ranking で返す
- standard graph は current session を強制し、model 指定の session id を信用しない
- graph traversal は session scope、predicate filter、depth / hit cap、stable order を持つ
- provenance は参照先 raw も session scope と件数上限を再検証する
- timeline search は model-visible path では current session だけを扱う
- string ではなく struct で返し、LLM 直前で整形する

federation MVP の scope 外:

- write federation
- replication
- auth / entitlement enforcement
- source discovery / registry integration

## 13. distillation 設計

distillation は「圧縮」ではなく「構造化」である。

入力:

- current loop の RawNode 群
- activated AbstractNode 参照

出力:

- 新規 AbstractNode
- raw lifecycle update

crate 本体は `Distiller` trait を公開し、example/test support に deterministic baseline 実装を置く。

- session / loop entity を立てる
- raw ノードとの relation を張る
- tool result relation を張る
- activated abstract に `informed_by` relation を張る
- provenance raw ids を relation に埋める

distillation 成功後は raw を `Distilled` に更新し、overflow bias を落とす。

commit 前には `(session_id, loop_id)` の distillation lease / fence を取得する。
claim は入力 raw ID digest、lease ID、expiry を持つ。distiller の実行後も claim
が current であることと、raw lifecycle update が入力集合の subset であることを
検証してから commit する。期限切れ・lease 喪失 worker は stale output を commit
できず、入力 digest 由来 operation key により crash retry も deduplicate される。

## 14. maintenance pass

`run_turn` は current loop の distillation を同期で行う。一方、session から押し出された backlog raw は `run_maintenance_pass` で後処理する。

maintenance の挙動:

- undistilled かつ pushed-out raw を bounded batch で取得
- `(session_id, loop_id)` 単位で group 化
- loop ごとに `max_loop_nodes` 以内の入力を claim して timeout 付きで distill
- 新規 abstract を保存
- fence を再確認してから入力集合内の raw lifecycle を更新

これにより session context からあふれた raw も後から graph memory に昇格できる。

crate 内の `run_maintenance_pass` は library API であり、定期実行の主体は
この crate では持たない。service wrapper / control-plane / worker scheduler が
run lifecycle に合わせて呼び出す境界である。

## 15. 設定

設定は `EngineConfig` に集約し、`toml` から読める。

主要セクション:

- `system_prompt`
- `memory`
- `context_budget`
- `tools`
- `runtime`

runtime では step budget、tool budget、timeout、maintenance batch size を管理する。

```toml
system_prompt = "You are the Rust-based Takos agent engine."

[memory.activation.target_ratio]
raw = 1
abstract = 1

[memory.activation]
top_k_total = 20
use_time_decay = true
overflow_raw_threshold_relaxation = true

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

config validation は reserve 合計・ratio arithmetic・threshold finite/range・timeout・collection /
payload / step 上限を checked arithmetic で fail-closed に検査する。runtime override も
config の hard ceiling を越えられない。

## 16. エラーと観測性

エラーは少なくとも次に分ける。

- configuration
- storage
- model
- tool
- indeterminate side-effecting tool outcome
- timeout
- cancellation / terminated loop
- unsafe recovery / checkpoint identity mismatch

runtime は `tracing` span を使い、少なくとも次を追えるようにする。

- `session_id`
- `loop_id`
- current node
- graph steps
- tool rounds
- maintenance batch

## 17. テスト戦略

現在のテストは deterministic backend を前提に組む。

- config parse / validation
- scoring
- context budget
- graph branching / timeout / paused resume
- checkpoint schema / graph / session / loop identity rejection
- side-effecting tool timeout / cancellation ambiguity と idempotent recovery
- session-scoped vector / graph / provenance / timeline query
- bounds / overflow / oversized payload rejection
- idempotent persistence
- object WAL replay / receipt uniqueness / cross-process lock / symlink boundary / quarantine
- maintenance pass、distillation lease / stale fence、lifecycle

CI は stable で `cargo fmt --all --check`、default/no-feature library check、all-target/all-feature clippy・test、
`RUSTDOCFLAGS="-D warnings" cargo doc` を実行する。別 job で MSRV Rust 1.85
の all-target/all-feature check を行い、すべて committed `Cargo.lock` を
`--locked` で使う。GitHub Actions は tag ではなく commit SHA に pin する。

実 model に依存しないため、CI で安定して再現できることを優先する。

## 18. 現時点の到達点と境界外の統合課題

到達点:

- library-first な stateless core
- graph runtime と checkpoint / paused resume
- bounded multi-step tool loop
- idempotency context と indeterminate side-effect boundary
- journal / receipt / advisory-lock 付き object persistence
- RawNode / AbstractNode 二層 memory
- session-scoped、overflow-aware retrieval
- lease / fence 付き bounded distillation
- object-backed 永続 backend
- feature-gated OpenAI-compatible embedding backend

crate 外の統合課題:

- 分散 scheduler
- network filesystem / multi-host backend の distributed transaction・lease
- multi-agent memory federation の write / replication / auth / source discovery

## 19. 一文で要約

`takos-agent-engine` は、session と長期 memory を同一 substrate
上で扱い、RawNode / AbstractNode の二層記憶を activation しながら、paused
checkpoint から再開できる graph runtime で長期継続実行する Rust agent engine
である。
