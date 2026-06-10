# Agent Runtime

> このページでわかること: Takos agent runtime の実装境界と Rust container の役割。

::: tip Status このページは current implementation の agent runtime 境界を説明します。Takos は control plane 全体を Rust
に寄せる方針ではなく、container 内の agent 本体を Rust の正本にする方針です。 :::

## 方針

Takos の agent 系で `all Rust` と呼ぶ対象は、`takos-agent` container の内側です。

Rust runtime が持つもの:

- agent loop orchestration
- memory substrate
- context assembly
- managed skill runtime copy
- skill catalog 合成と activation
- prompt construction
- local memory tools
- model runner contract

Workers / control plane 側に残すもの:

- run queue と run lifecycle 管理
- auth / billing / space / thread / run state
- remote tool catalog と tool 実行実体
- custom skill の CRUD と永続化
- executor-host の host process

この分離が current canonical architecture です。

`takos-agent-engine/` 単体の tool surface は `executor` と `memory_tools` です。\
`skill_list` / `skill_get` / `skill_context` / `skill_catalog` / `skill_describe` の local intercept は `takos/containers/agent/`
側の wrapper が担い、managed skill と custom skill の合成結果を返します。

## 実行構成

```text
control-web / control-worker
  -> executor-host
     -> takos-agent container
        -> takos-agent-engine (engine core / memory tools)
        -> wrapper-side skill intercept
        -> local memory tools
        -> remote tool bridge
             -> control RPC
             -> Workers / platform tools
```

`takos-agent` は container の inside loop を責務として持ち、Takos product control state は control plane に委譲します。

## なぜこの分離か

- agent の思考ループは Rust で型安全に固定したい
- tool backend と Takos product control state は Takos 本体の Workers/DB と密結合している
- custom skill や remote tool を全部 Rust に移すと、product control plane の変更速度を落とす
- 一方で container 内の loop を Rust にすれば、agent 自体の信頼性と再現性は高められる

つまり、Takos における Rust 化の目的は「product control plane を全部書き換えること」ではなく、「agent container の本体を Rust
の正本にすること」です。

## Local と Remote の境界

`takos-agent` は local tool と remote tool を明示的に分けます。

engine core local:

- `semantic_search_memory`
- `graph_search_memory`
- `provenance_lookup`
- `timeline_search`

wrapper-side local skill intercept:

- `skill_list`
- `skill_get`
- `skill_context`
- `skill_catalog`
- `skill_describe`

`skill_*` 系は `takos-agent-engine` の公開 tool surface ではなく、`takos/containers/agent/` の wrapper が local intercept します。

remote:

- repo / file / deploy / runtime / MCP / space などの platform tool

同名の tool がある場合は Rust container の local 実装が優先です。\
remote tool の実体は control plane が持ち、Rust 側は catalog と execution を RPC で扱います。

## Skill の正本

managed skill:

- Rust 側は runtime copy を保持
- TS control-plane 側も skill API / source 定義を持つため、完全な単一正本ではない

custom skill:

- control plane DB が正本
- Rust 側は runtime ごとに catalog に取り込み、selection と prompt 化を行う

このため、「agent core の振る舞い」は Rust 側に寄せつつ、space 管理データと custom skill 永続化は control plane
正本のままです。

## 実装上の source of truth

- agent core: standalone `takos-agent-engine/` の `Cargo.toml` と `src/`
- service wrapper: `takos/containers/agent/` は bootstrap / control RPC / prompt / skill wiring を担う wrapper
- control RPC contract: Takos control plane 側の container-hosts / executor 間 RPC 定義

Takos-agent の Rust 実装は、engine core を standalone `takos-agent-engine/` に置き、ecosystem checkout 側では
`takos/containers/agent/` service wrapper から path dependency として参照する。
