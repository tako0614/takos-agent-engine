# Agent Runtime

> Takos agent runtime の current 実装境界と Rust executor の役割。

## Authority

Takos Worker が product state の唯一の durable authority です。

- Thread message、summary、explicit memory、retrieval indexes
- Run lifecycle、cancel、lease、usage、event
- managed/custom skill catalog
- static tool policy、installed Capsule / external MCP discovery、tool execution

`takos-agent` container は run-scoped executor です。

- bounded graph loop と cancellation
- product から渡された structured history の model context 化
- provider adapter と native tool-call / tool-result correlation
- remote tool bridge
- run-scoped checkpoint (product recovery authorityではない)

container の disk、pool slot、process lifetime は product state の正本ではありません。sleep / restart / 別 slotでも、次の
run は Takos Worker の canonical history から再構築します。`takos-agent-engine` の file-backed repository と
`resume_loop` は library consumer が durable backendをinjectする場合のprimitiveであり、Takos wrapperのcrash recoveryを
意味しません。

Takos wrapperは`ExecutionProfile::ExternalContext`を明示します。このprofileはengineのmemory-aware defaultを他consumer向けに
残したまま、ingest / session reload / local embedding・activation / tool-result persistence / overflow / distillation nodeを通らない
5-node model/tool loopを選びます。Worker history、current user、current-turn tool transcriptはそれぞれ専用のstructured fieldでmodelへ
一度だけ渡し、engineのsession/memory string contextへ複製しません。graph/tool budget、timeout、cancellation、checkpoint、native
tool-call ID、`SessionResponse.turn_messages`はmemory-aware profileと共通です。

## Runtime flow

```text
Takos Worker
  ├─ queue / Run state / lease / cancellation
  ├─ canonical Thread history + retrieved context
  └─ tool catalog + authorization + execution
           │ run-scoped agent-control RPC
           ▼
takos-agent container
  ├─ provider-neutral structured transcript
  ├─ model adapter
  ├─ bounded takos-agent-engine graph
  └─ remote tool bridge
           │ correlated tool_call_id
           └──────────────► Takos Worker
```

Takosumi は Capsule / ContainerService のdeploy、credential、Run / StateVersion / Output / audit boundaryを管理します。
Takos固有のconversation、memory、skill、tool-control RPCはTakos Workerが所有します。

## Tool boundary

Model-visible catalogの正本はTakos Workerです。

- Takos core toolは小さいstatic catalogとしてWorkerに残す
- computer / Git / general file / storage / Web searchはinstalled Capsuleまたはexternal MCPからdiscoverする
- `toolbox`が選択的catalogの入口になる
- authorization、side-effect classification、idempotencyはWorkerで強制する

engineの`semantic_search_memory` / `graph_search_memory` / `provenance_lookup` / `timeline_search`はmemory-aware library
consumer向けprimitiveです。Takos production wrapperはexternal-context profileを使うため、これらをlocal product memory authorityや
turn-local duplicate memoryとして実行しません。

## History and model protocol

Workerのconversation-history responseは`system` / `user` / `assistant` / `tool` role、`tool_calls`、`tool_call_id`を保持した
provider-neutral transcriptへnormalizeします。current user messageは`SessionRequest`、過去履歴は`RunOptions`でengineへ渡し、
重複させません。

Takosのcanonical `tool_calls` wire shapeはflatな`{ id, name, arguments }`です。OpenAIのnested `function` shapeはprovider
adapterの内側だけで扱い、Rust history readerは保存済み移行データを読む期間に限ってlegacy nested shapeも受理します。
history trimはassistant callと対応する全tool resultを不可分単位にし、orphan/incomplete exchangeをproviderへ送りません。

model adapterはassistant tool call IDを保存し、同じIDをtool execution、event、tool-result messageへ通します。parallel callを
nameや配列順だけで相関しません。

`ToolExecutor::execution_kind` は各callを read-only / side-effecting に分類します。engineは隣接するread-only callだけを
parallel実行し、side-effecting callをprovider順のbarrierとして直列実行します。未分類はside-effecting扱いでfail-closedに
します。

tool-result contentはmemory-aware / external-contextの両profileでcurrent turn合計`reserve_tools`以内にclampします。correlation
messageは落とさず、超過内容をstructured previewへ変換します。完全なlarge outputはtool実装がartifact/objectへ保存し、modelには
bounded preview/referenceを返す責務です。

## Source owners

- engine library: `takos-agent-engine/`
- Takos executor wrapper: `takos/containers/agent/`
- product agent-control RPC / durable state / tools: `takos/src/worker/`
- Capsule / ContainerService deployment and OpenTofu Run ledger: `takosumi/`

## Checks

```bash
cd takos-agent-engine
cargo test --features test-support
cargo fmt --check
cargo clippy --all-targets --features test-support -- -D warnings

cd ../takos/containers/agent
cargo test --features mock-llm
cargo fmt --check
cargo clippy --all-targets --features mock-llm -- -D warnings
```
