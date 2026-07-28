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

durable recoveryを使うlibrary consumerは`RunOptions.loop_id`にcaller-stable IDを渡します。checkpointはschema version、
profile固有`graph_id`、session / loop identityを照合するため、別graphや別runの状態を誤って再開しません。Takos wrapperは
引き続きWorker canonical historyからrunを再構築し、container checkpointをproduct authorityにはしません。

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

memory-aware consumer向けの4 memory toolはstandard graphがcurrent sessionを強制します。modelが別sessionをargumentsへ
入れても上書きし、vector / graph / provenance / timelineの各層でscopeを再検証します。

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
します。call数・ID / name・argumentsはdispatch前に検証し、上限超過を黙ってdropしません。

wrapperは`ToolExecutor::execute_with_context`で渡される`session_id`、`loop_id`、`idempotency_key`、timeout、
cancellation tokenをWorker control RPCまで伝播します。Workerのdurable tool operationはengine keyをconditional
create / lookupのfenceとして使い、同じkeyの再送で副作用を二重実行しません。`recovery_is_idempotent`は、このend-to-end
fenceが実在するcallにだけtrueを返します。

read-only callのtimeoutは通常のtool errorとしてmodelへ返せます。side-effecting callはdispatch後にtimeout /
cancellationになってもremote commitの有無をengineから判定できないため、`ToolOutcomeIndeterminate`でrunを停止します。
wrapperはこの状態を一般的な失敗へ潰さず、idempotency keyとともにWorkerへ返します。子process / HTTP requestの中止・reapは
executor実装の責務です。

tool-result envelopeはmemory-aware / external-contextの両profileで`max_tool_result_bytes`以内へ縮約し、current turn全体も
`reserve_tools`以内にclampします。correlation messageは落とさず、超過内容をstructured previewへ変換します。engine local
storeにfull resultを残す前提にはしません。完全なlarge outputはWorker/tool実装がdurable artifact/objectへ先に保存し、
modelにはbounded preview/referenceを返します。

## Source owners

- engine library: `takos-agent-engine/`
- Takos executor wrapper: `takos/containers/agent/`
- product agent-control RPC / durable state / tools: `takos/src/worker/`
- Capsule / ContainerService deployment and OpenTofu Run ledger: `takosumi/`

## Checks

```bash
cd takos-agent-engine
cargo fmt --all --check
cargo check --locked --lib --no-default-features
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --locked --no-deps --all-features
cargo +1.85.0 check --locked --all-targets --all-features

cd ../takos/containers/agent
cargo test --features mock-llm
cargo fmt --check
cargo clippy --all-targets --features mock-llm -- -D warnings
```
