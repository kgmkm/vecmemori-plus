<h1 align="center">vecmemori-plus</h1>
<p align="center">
  <em>AIエージェントのためのローカル事実記憶 — SQLite + FTS5 + ニューラル埋め込み + マルチエージェント共有MCPサーバー</em>
  <br>
  <a href="README.md">English README</a>
</p>

---

**vecmemori-plus** は、AIエージェント向けのローカル・永続的な事実記憶エンジンです。
(import モジュール名は互換性のため `vecmemori` のまま。pip パッケージ名は `vecmemori-plus` です。)ユーザーの好み、プロジェクトの決定事項、環境情報などの短い事実を SQLite に保存し、次の2系統で関連する記憶を検索します。

- **FTS5全文検索** — キーワード一致の精度を担当
- **ニューラル埋め込み** — 意味的な近さを担当

埋め込みは中核機能です。vecmemori-plus は「単なる全文検索DB」ではなく、意味検索を前提としたメモリシステムです。

> **ステータス:** Alpha。コアAPIは利用可能ですが、安定版まではパッケージングや Hermes Agent 連携方法が変わる可能性があります。

## このフォークについて

このリポジトリは [`iwaan10000vr/vecmemori`](https://github.com/iwaan10000vr/vecmemori) のフォークです。日本語ローカルメモリエンジンを生み出した [@iwaan](https://github.com/iwaan10000vr) に感謝します。コア設計(SQLite + FTS5 + ニューラル埋め込み、fugashi による分かち書き、デフォルトモデルとしての ruri-v3)の功績はすべて上流プロジェクトに帰属します。

**フォークの意図。** 私たちは長編小説プロジェクトを複数のAIエージェント(Hermes Agent、opencode、goose、OpenClaw)で運用しており、**全エージェントから1つの共有メモリDB**を参照したい — 同じキャラ設定・世界観・伏線がどのエージェントからも見える状態 — を目指しました。上流は Hermes Agent 専用の統合で、2026年5月以降更新が静かだったため、待つのではなく拡張版をこのリポジトリで保守することにしました。

**このフォークが追加したもの(v0.2.0):**

- **MCP stdioサーバー**(`vecmemori.mcp_server`)— `fact_store`(9アクション)と `fact_feedback` を任意のMCP対応エージェントに公開。各エージェントの設定1行+共通SQLite DB 1つで接続。詳細は [MCPサーバー](#mcpサーバーマルチエージェント共有メモリ)。
- **日本語FTS5のAND罠の解消** — 自然文クエリをトークナイズし、日本語の助詞を除去、OR展開+引用符化。全文クエリでも recall が0にならない。
- **`FactRetriever.search()` の `fts_query` パラメータ** — FTS5 MATCH 式を埋め込みクエリから分離。クエリ展開が意味検索経路を汚染しない。
- **`contradict` の改善** — 陳述に類似するファクトを検索するようになりました(上流は最近のファクト一覧を返すだけのスタブ)。断定前に確認できる。
- **FTS専用モード**(`VECMEMORI_FTS_ONLY=1`)— 1.2GBの埋め込みモデルなしで動作。軽量エージェント向け。
- **環境変数による設定** — `VECMEMORI_DB`、`VECMEMORI_FACT_LANGUAGE` など。MCP経路に Hermes 依存なし。
- **opencode / goose / OpenClaw の設定スニペット(接続テスト済み)** と stdio E2E テスト(`tests/test_mcp_e2e.py`)。

Hermes Agent 用プロバイダアダプタは挙動を変えずに維持しています。

## プライバシーモデル

コアライブラリは SQLite とローカルの SentenceTransformer 互換モデルを使います。この構成ではデータは端末内に留まります。

ただし、統合機能によっては設定済みの LLM プロバイダーにテキストを送る場合があります。

- Hermes Agent の `auto_extract` は、会話履歴を LLM に送り、保存すべき事実を抽出することがあります。
- Hermes Agent の `retrieval_planner` / planner injection は、検索クエリ生成のために会話文脈を LLM に送ることがあります。
- デフォルト外のリモート埋め込みプロバイダーを明示的に設定した場合、埋め込み対象テキストが外部に送られる可能性があります。

利用する LLM/埋め込みプロバイダーを信頼できる場合のみ、それらの機能を有効にしてください。

## インストール

```bash
pip install vecmemori-plus
pip install vecmemori-plus[ja]       # fugashi + unidic-lite による日本語FTS5対応
pip install vecmemori-plus[hermes]   # Hermes Agent 環境向けアダプター依存
pip install vecmemori-plus[all]      # 日本語 + Hermes アダプター依存
```

事実の保存・検索の前に、ローカル埋め込みモデルを用意します。

```bash
bash scripts/download_model.sh
```

デフォルトのモデル配置先:

```text
~/.cache/vecmemori/models/ruri-v3-310m
```

デフォルトモデルは [`cl-nagoya/ruri-v3-310m`](https://huggingface.co/cl-nagoya/ruri-v3-310m) です。Apache-2.0 ライセンスで、日本語に強いモデルです。

## クイックスタート

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
store.add_fact("ユーザーはダークモードを好む", category="user_pref")
store.add_fact("プロジェクトでは pytest を使う", category="project")

retriever = FactRetriever(store=store)
results = retriever.search("テスト環境と好み", limit=5)

for r in results:
    print(f"[{r['trust_score']:.2f}] {r['content']}")
```

埋め込みなしの明示的なテスト・診断だけは、`MemoryStore` と `FactRetriever` の両方に `require_embeddings=False` を渡すことで実行できます。本番利用では推奨しません。

## 日本語FTS5対応

SQLite 標準の `unicode61` tokenizer は日本語をうまく分割できません。`vecmemori-plus[ja]` を入れると [fugashi](https://github.com/polm/fugashi) / MeCab で事前分かち書きしてから FTS5 に登録します。

```bash
pip install vecmemori-plus[ja]
python -c "from vecmemori._tokenizer import has_tokenizer; print('Japanese FTS5:', has_tokenizer())"
```

意味検索の主軸はニューラル埋め込みです。日本語FTS5は、キーワード一致の精度を補強する役割です。

## 特徴

- **2系統検索** — FTS5 + ニューラル埋め込み。デフォルト重みは 0.40 / 0.60
- **埋め込み前提** — ローカル SentenceTransformer 互換モデルを使う
- **日本語対応** — ruri-v3 埋め込み + fugashi による任意のFTS5形態素解析
- **エンティティ抽出** — 事実文から軽量に固有表現を抽出
- **信頼度スコア** — helpful/unhelpful フィードバックで信頼度を調整
- **時間減衰** — 古い事実を弱めるオプション
- **モデル差し替え** — 埋め込み次元、query/doc prefix を設定可能
- **Hermes Agent連携** — memory provider、`fact_store` tool、prefetch、任意のLLM planner/extraction

## Python API

```python
from vecmemori import MemoryStore, FactRetriever

store = MemoryStore(db_path="memory.db")
fact_id = store.add_fact("GPU: RTX 5060 Ti 16GB", category="tool", tags="hardware,gpu")
store.record_feedback(fact_id=fact_id, helpful=True)

retriever = FactRetriever(store=store)
results = retriever.search("GPUメモリ", limit=5)

store.rebuild_all_embeddings()  # 埋め込みモデル/設定変更後に再生成
store.close()
```

`probe`, `reason`, `contradict` は standalone の `FactRetriever` メソッドではなく、Hermes の `fact_store` tool action です。

## Hermes Agent連携

`vecmemori-plus[hermes]` は Hermes plugin module に必要な `PyYAML` / `httpx` などを入れます。Hermes Agent本体はインストールしません。既に Hermes Agent が入っている環境に追加するための extra です。

### GitHub から Hermes Agent に入れる

専用の Hermes plugin installer を整えるまでは、source checkout を plugin directory として使い、Hermes が動く Python 環境に vecmemori をインストールしてください。

```bash
git clone https://github.com/iwaan10000vr/vecmemori.git
cd vecmemori
python -m pip install -e ".[hermes,ja]"
bash scripts/download_model.sh

mkdir -p ~/.hermes/plugins
ln -s "$(pwd)/src/vecmemori/hermes" ~/.hermes/plugins/vecmemori
hermes config set memory.provider vecmemori
hermes config set plugins.vecmemori.embedding_model "$HOME/.cache/vecmemori/models/ruri-v3-310m"
```

任意のローカル優先設定:

```bash
hermes config set plugins.vecmemori.auto_extract true
hermes config set plugins.vecmemori.fact_storage_language ja   # 保存するfact本文の言語
hermes config set plugins.vecmemori.retrieval_planner false
hermes config set plugins.vecmemori.embedding_trust_remote_code false
```

plugin code や memory provider 設定を変えた後は、Hermes Agent を再起動してください。

Hermes memory provider として使う場合の流れ:

1. **毎ユーザーメッセージ:** prefetch が関連factを検索し、上位候補を文脈に注入
2. **tool call:** `fact_store` で fact を明示的に追加・検索・更新・削除
3. **memory mirroring:** Hermes標準の `memory` tool 書き込みを vecmemori にも反映
4. **session end:** 任意で `auto_extract` が LLM に durable fact 抽出を依頼
5. **planner:** 任意で `retrieval_planner` が LLM に複数検索クエリ生成を依頼

Hermes専用 tool action:

- `add`, `search`, `probe`, `related`, `reason`, `contradict`, `update`, `remove`, `list`

`reason` / `contradict` は実用的な意味検索補助であり、数学的な代数推論や完全な矛盾検出を保証するものではありません。

## MCPサーバー(マルチエージェント共有メモリ)

`vecmemori.mcp_server` モデルは、同じファクトメモリを **任意のMCP対応エージェント**(opencode、goose、OpenClaw など)に公開します。

> **接続テスト済み**: opencode 1.18 / goose 1.49 / OpenClaw 2026.9(書き込み+クロスエージェント読み取り+feedback を単一共有DBで検証)。全エージェントが1つのSQLiteデータベース・1つの埋め込みモデルを共有します。

### 起動

```bash
pip install -e ".[ja]"
bash scripts/download_model.sh   # 意味検索を使う場合(任意)

# 埋め込みなし(FTS専用)で素早く試す:
VECMEMORI_FTS_ONLY=1 python -m vecmemori.mcp_server
```

環境変数:

| 変数 | デフォルト | 意味 |
|---|---|---|
| `VECMEMORI_DB` | `~/.vecmemori/memory.db` | SQLiteパス(**全エージェントで共通化**) |
| `VECMEMORI_EMBEDDING_MODEL` | `~/.vecmemori/models/ruri-v3-310m` | ローカル埋め込みモデル |
| `VECMEMORI_FTS_ONLY` | off | `1` で埋め込みを無効化(FTS5のみ) |
| `VECMEMORI_FACT_LANGUAGE` | 未設定 | 保存言語ガード(`ja`/`en`/`ko`/`zh`) |
| `VECMEMORI_MIN_TRUST` | `0.3` | 検索の最低trust |

公開ツール: `fact_store`(アクション: add / search / probe / related / reason / contradict / update / remove / list)、`fact_feedback`(helpful / unhelpful)。

### 各エージェントの設定

**opencode**(`~/.config/opencode/opencode.json`):

```jsonc
{
  "mcp": {
    "vecmemori": {
      "type": "local",
      "command": ["C:/Users/<you>/path/to/python.exe", "-m", "vecmemori.mcp_server"],
      "enabled": true,
      "environment": {
        "VECMEMORI_DB": "C:/Users/<you>/.vecmemori/memory.db",
        "VECMEMORI_FACT_LANGUAGE": "ja"
      }
    }
  }
}
```

**goose / OpenClaw** — 同じ stdio コマンド(`python -m vecmemori.mcp_server`)をMCPサーバーとして登録。goose は `goose configure`、OpenClaw は Control UI → Settings → MCP(または config の `mcp.servers`)。

> **注意**: Hermes Agent 内では引き続きネイティブプロバイダ(自動prefetch+LLMプランナ)を推奨。MCPサーバーは他エージェント用で、同じ `VECMEMORI_DB` を共有できます。

### contradictアクション

`contradict` は陳述に意味的に近いファクトを検索し、**断定前に候補を確認させる**ためのアクションです。論理推論ではなくrecallベースのヒューリスティックです。返された候補を人間(またはエージェント)が比較判断します。

## 設定

主な設定:

- `db_path`: SQLite DB パス。standalone ではデフォルト `memory.db`
- `default_trust`: 新規factの初期信頼度。デフォルト `0.5`
- `fts_weight`: キーワード検索スコアの重み。デフォルト `0.40`
- `ruri_weight`: 意味埋め込みスコアの重み。デフォルト `0.60`（互換性のため旧キー名を維持）
- `prefetch_limit`: Hermesで毎ターン注入するfact数。デフォルト `5`
- `auto_extract`: Hermesでセッション終了時にLLM抽出を行うか
- `fact_storage_language`: 保存するfact本文の言語（例: `ja`, `en`）。未設定/null/auto の場合はfact書き込みを拒否し、先にどの言語で保存するかユーザーへ確認する。固有名詞、コード、パス、製品名、自然な専門用語は原語表記を保持する。
- `retrieval_planner`: HermesでLLMによるマルチクエリ検索を使うか
- `embedding_model`: ローカル埋め込みモデルパス
- `embedding_trust_remote_code`: Hugging Face custom model code を許可するか。デフォルト `false`

## 埋め込みモデルの差し替え

```python
from vecmemori._embedder import set_config, set_model_path

set_config(dimension=384, query_prefix="", doc_prefix="")
set_model_path("/path/to/all-MiniLM-L6-v2")

from vecmemori import MemoryStore
store = MemoryStore("memory.db")
store.rebuild_all_embeddings()
```

推奨例:

- `cl-nagoya/ruri-v3-310m` — 日本語、768次元、約1.2GB
- `sentence-transformers/all-MiniLM-L6-v2` — 英語、384次元、小型
- `BAAI/bge-large-en-v1.5` — 英語、1024次元
- `intfloat/multilingual-e5-large` — 多言語、1024次元

## アーキテクチャ

```text
ユーザークエリ
    │
    ▼
┌──────────────────────────┐
│ 1. FTS5 / BM25           │  weight=0.40 — キーワード一致
│ 2. ニューラルcos類似度   │  weight=0.60 — 意味的近さ
└──────────┬───────────────┘
           ▼
    関連度 × 信頼度 × 時間減衰
           ▼
    ソート結果
```

## 開発

```bash
pip install -e ".[dev,ja]"
python -m pytest -q
python -m build --sdist --wheel
python -m twine check dist/*
check-manifest
```

## ライセンス

MIT — [LICENSE](LICENSE) を参照。

第三者ライセンス情報は [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) にまとめています。Hermes Agent / Holographic memory plugin 由来部分の attribution は [NOTICE](NOTICE) と third-party notices に記載しています。

## 謝辞

**vecmemori-plus** は、上流 [`vecmemori`](https://github.com/iwaan10000vr/vecmemori) の作者である [@iwaan](https://github.com/iwaan10000vr) に大きく負っています。このフォークはその成果の上に成り立っています — コアのストア/レトリーバ設計、fugashi による日本語FTS5のアプローチ、デフォルト埋め込みモデルとしての cl-nagoya の ruri-v3 採用は、すべて上流から来ています。きれいで読みやすい土台をありがとうございます。

さらに遡ると、上流 vecmemori 自体が Nous Research の [Hermes Agent](https://github.com/NousResearch/hermes-agent) に同梱されていた MIT ライセンスの `holographic` memory plugin のフォーク/再実装として始まりました。このリポジトリはその系譜の第3世代です。

Hermes Agent、SQLite、NumPy、sentence-transformers、PyTorch、fugashi/MeCab、UniDic、cl-nagoya の ruri モデルの開発者・メンテナーにも感謝します。