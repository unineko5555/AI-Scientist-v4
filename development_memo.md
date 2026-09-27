# V2の研究機能を維持したCodex CLI / Claude Code CLI対応仕様

作成日: 2026-09-27

対象: AI-Scientist-v4（このリポジトリに存在するV2由来の実装）

調査基準: `6699ce614b8a8847d690118f3dff7e9ab6a5cb15`

状態: **共通CLI推論層とV2各入口を実装。オフライン回帰とCodex小規模実測を実施中。Claude実機・CUDA/LaTeXを含むV2全工程の受入は未完了。** 最新の実装差分・検証結果・残条件は17節を参照。1〜16節の「現状」および行番号は初版調査時点を指す。

## 1. 結論と変更しないもの

採用する方針は、**V2のPythonオーケストレーターを残し、LLM/VLMへの推論経路だけを共通化して、その下にCodex CLIとClaude Code CLIを追加すること**である。

CLIに「研究を全部実行して」と依頼する方式にはしない。研究段階、探索木、実験実行、指標抽出、画像評価、論文生成、査読を管理する主体は引き続きV2であり、CLIは原則として「指定された入力に対して一つの回答を返す推論プロセス」として扱う。

`llm.py`のクライアント生成だけを差し替えても完成しない。独立したtree-search backend、VLM、SDK直接呼び出し、モデル名による分岐、プロセスをまたぐ並列実行、利用量計測まで接続し直す必要がある。

### 1.1 必須要件

| ID | 要件 |
|---|---|
| R1 | V2の研究機能を削除せず、Codex CLIだけ、Claude Code CLIだけの双方で実行できること。両者の役割別併用も可能にする。 |
| R2 | `subscription_only`モードでは、本体が管理する全LLM/VLM呼び出しを指定CLIへ送り、API課金経路へ自動fallbackしない。 |
| R3 | 既存APIモードと既存の公開helperの戻り値形式を維持する。backendとmodelを別々に設定する。 |
| R4 | 画像を実際に入力し、構造化応答、会話履歴、複数回答、reflectionを維持する。未対応機能をskipや成功扱いで代替しない。 |
| R5 | CLIの認証切れ・利用上限・timeoutを、実験のバグや不正なグラフという科学的判定に変換しない。 |
| R6 | 子プロセス・一時ファイル・履歴をリクエストごとに分離し、停止時にも完了済み実験と論文の成果を保持する。 |
| R7 | provider、要求モデル、判明した実モデル、CLIバージョン、設定、失敗理由、取得可能なusageを追跡できること。 |

「V2維持」は、工程・入出力契約・探索と評価の意味・成果物を維持するという意味である。モデル変更による研究品質の差や、APIとCLIのsampling/role semanticsの差まで同一になるとは保証しない。これらの差は隠さず記録する。

### 1.2 今回の対象外

- V3へのコードベース置換、Harborへの移行、BFTS/AgentManager/Journalの削除。
- CLI自身に実験・shell実行・LaTeX修正を任せる新しい研究エージェント設計。
- OAuthトークン抽出、非公開APIの呼び出し、サブスク認証を流用した独自APIサーバー。
- 無制限利用・必ず追加費用ゼロという保証。契約枠、追加usage、GPU、データ取得等の費用は別問題。
- 最初のCLI対応と同時に、任意の処理位置から完全再開できるcheckpointシステムを新設すること。

Semantic Scholar、データセット取得、GPU/PyTorch、PDF/LaTeX処理等の既存依存は残る。「LLM用APIキーなしで動く」と「外部サービスなし・完全offline」は異なる。

## 2. 調査範囲と確度

ローカルの全Pythonファイル36本・13,213行に加え、設定、README、テンプレート、可視化、例示データを分担読解した分析を基礎とする。以下の行番号は調査時点の位置であり、実装時は関数名を優先して参照する。

| 対象 | 確認内容 | 未確認事項 |
|---|---|---|
| このリポジトリ | 呼び出し経路、設定伝播、戻り値、画像・査読・実験の依存関係を静的確認 | 学習、論文生成、API推論を伴う実行検証は未実施 |
| ローカルCodex | `codex --version`、`exec --help`等。インストール版は`0.157.1` | ログイン状態、契約、実モデルの利用可否、画像/schemaの実推論は未確認 |
| ローカルClaude | `claude`がPATH上にないこと | インストール、認証、コマンド実行、実推論は未実施 |
| CLI仕様 | OpenAI/Anthropic公式ドキュメント、Anthropic公式SDKのCLI接続実装 | バージョン依存の組み合わせ動作はPhase 0で検証する |
| V3 | 指定repositoryの下記固定コミットを隔離ディレクトリで静的調査 | Harborを含む実運用・サブスク認証・テストの実行は未実施 |

V3参照コミット: [`b03021d78215271e5a5b2a894db6d3fadbe7a06c`](https://github.com/findalexli/ai-scientist-v3/tree/b03021d78215271e5a5b2a894db6d3fadbe7a06c)。

本書では「現状」はコードから確認できた事実、「採用仕様」は今回提案する設計、「検証ゲート」は実装前後に実測する条件として区別する。現行CLIの詳細は変化するため、対応版と検証結果を固定して管理する。

## 3. V2の維持対象と現在の推論経路

### 3.1 維持する研究パイプライン

```text
アイデア生成 / 既存アイデア選択
  → AgentManagerの4研究段階
      初期実装 → baseline調整 → creative research → ablation
      └ 各段階: 木探索 / debug / improve / 指標抽出 / plot / VLM / node選択
  → multi-seed実験・集約
  → Journal / JSON要約 / HTML可視化
  → 集約plot・文献検索・引用追加
  → 通常writeup または ICBINB writeup
      └ 図説明 / reflection / 図重複・掲載判断 / ページ数調整 / LaTeX・PDF
  → 論文テキスト査読 + 図・caption・本文参照の最終査読
```

この図は機能の関係を示すもので、全てが単純な一方向処理という意味ではない。実際の探索、修正、reflectionは反復する。標準launcherはICBINB系を使用するが、通常writeupの独立経路も維持対象とする。

図の重複検出・掲載判断等はICBINB側にある処理であり、CLI移行に合わせて通常writeupへ新たに追加する要件ではない。各形式の既存の機能境界を維持する。

特にVLMは付加的なレビューではない。`Journal.good_nodes`は実験とplot双方の成功を要求するため、探索中のVLM結果は候補選択、成功データセット、段階終了にも影響する。VLMを省くとV2と異なる探索になる。

### 3.2 推論入口は二系統と一つの直接呼び出し

以降、特記のないPythonパスは`ai_scientist/`配下。

| 修正入口 | 現状と重要な契約 | CLI対応の変更 |
|---|---|---|
| `llm.py:create_client`（480行付近） | OpenAI、Anthropic、Bedrock、Vertex、OpenAI互換等のSDKを生成。`(client, model)`を返す | 明示backendから共通handleを作る。CLI分岐を既存model文字列判定より先に解決 |
| `llm.py:get_response_from_llm`（267） | 単発textと更新historyを返す。SDK別分岐とHugging Faceへの直接HTTP fallbackあり | text/history契約を維持し共通gatewayへ。CLIではSDK/HTTP fallbackに入らない |
| `llm.py:get_batch_responses_from_llm`（87） | native `n`または単発反復で`(texts, histories)`を返す | CLIでは独立したn要求に展開。順序・件数・history分岐を保持 |
| `vlm.py`（single:132、factory:195、batch:251） | 独立モデル一覧とOpenAI/Ollama呼び出し。画像encodeとbatchを持つ | textと同一gatewayを利用。実画像の添付をprovider別変換 |
| `treesearch/backend/__init__.py`（4、19） | model名に`claude-`があればBedrock、それ以外はOpenAI系。`query`は`str`または`dict` | model名でtransportを推測せず、明示backendへ。共通handleを利用 |
| `treesearch/backend/backend_openai.py`（31） | `FunctionSpec`をforced tool callに変換しargumentsをJSON parse | CLIではschema付き生成へ変換し、hostで検証したdictを返す |
| `treesearch/backend/backend_anthropic.py`（21） | Bedrock。現状`func_spec`未対応 | APIモードの仕様と区別し、新しいClaude CLI adapterではschema契約を実装 |
| `perform_vlm_review.py:detect_duplicate_figures`（389、SDK呼び出し433） | 複数画像を直接`client.chat.completions.create`へ送る | **必ずgatewayへ移す。`llm.py`だけの変更では残る迂回路** |

SDKの`isinstance`判定は確認されていない。一方でmodel名、static allowlist、clientの`.chat`/`.messages`、responseの`.usage`/`.choices`等への依存はある。偽のSDK互換オブジェクトを大量に実装するより、呼び出し境界で結果を正規化する方がよい。

### 3.3 呼び出し側の接続漏れを防ぐ一覧

| 機能・ファイル | 接続する推論用途 |
|---|---|
| `perform_ideation_temp_free.py` | アイデア生成と反復、独立CLI |
| `treesearch/parallel_agent.py` | code生成/debug/improve、execution review、dataset判定、plot選択とVLM、node summary、global metric、metric parse、tuning/ablation案、seed集約plot |
| `treesearch/agent_manager.py` | stage/substageのgoal、構成、終了・進行判定 |
| `treesearch/journal.py` | best node選択、running summary、実験ノート |
| `treesearch/log_summarization.py` | stage要約、summary更新、history注釈。tree側factoryのclientをtext側helperに渡す経路あり |
| `treesearch/journal2report.py` | 現行最終要約とは別の補助report生成機能も維持 |
| `perform_plotting.py` | 集約plotの生成・修正、独立CLI |
| `perform_writeup.py` | 通常writeup、引用、図説明、reflection、独立CLI |
| `perform_icbinb_writeup.py` | ICBINB執筆、引用、図説明・重複・選択、reflection、ページ制限、独立CLI |
| `perform_llm_review.py` | 単独査読、ensemble、meta-review、reflection |
| `perform_vlm_review.py` | 図説明、caption、参照、配置・情報量・重複判定 |
| `launch_scientist_bfts.py` | 各工程のclient/config伝播、最後のtext/VLM査読 |

現在のmodel指定は一か所に集約されていない。`bfts_config.yaml`のcode/feedback/vlm_feedback/reportに加え、optionalなsummary/select_node、`journal.py:471/544/607`、`log_summarization.py:272/344`の既定値も対象となる。

launcherの`model_writeup_small`は画像評価にも使用され、`model_review`はtextと画像の両方に使用される。新設するvision roleに明示上書きを認め、旧指定は互換aliasとして残す。

## 4. V3から採用するもの・採用しないもの

V3はV2のstage/BFS/journal/evaluatorを取り除く方針を明記している。したがって、V2との機能互換の見本ではなく、CLI実行の限定的な参考実装とする。[V3 README](https://github.com/findalexli/ai-scientist-v3/blob/b03021d78215271e5a5b2a894db6d3fadbe7a06c/README.md#L7-L19)

| V3の実装 | 今回の扱い |
|---|---|
| backendを明示選択する起動処理 | 採用する考え方。model文字列からAPI/CLIを推測しない |
| Codexの`exec --json` | イベント取得の参考。最終回答のJSON schemaを保証する機能とは混同しない |
| reviewerがstdinでpromptを渡し、`--output-last-message`で回答保存 | 採用する。大きな論文をargvに入れず、イベントと最終回答を分離 |
| Claude reviewerの`-p --output-format text` | 非対話呼び出しの参考。V2には画像・usage・構造化出力も必要なので、これだけでは足りない |
| periodicな成果物保存 | 失敗時にも成果を残す考え方を採用。V2のJournal保存をCLIログで置き換えない |
| CLIエージェントに研究作業全体を任せるHarbor構成 | 採用しない。V2のオーケストレーターを維持 |

CLIの具体例は[Codex adapter](https://github.com/findalexli/ai-scientist-v3/blob/b03021d78215271e5a5b2a894db6d3fadbe7a06c/local_harbor_agents/patched_codex.py#L132-L224)、[reviewerのCLI起動](https://github.com/findalexli/ai-scientist-v3/blob/b03021d78215271e5a5b2a894db6d3fadbe7a06c/scripts/submit_for_review.sh#L158-L212)を参照する。Claudeの主エージェント側コマンド構築はupstream Harborに委譲されており、このrepositoryだけに完全なClaude transport実装があるわけではない。[Claude wrapper](https://github.com/findalexli/ai-scientist-v3/blob/b03021d78215271e5a5b2a894db6d3fadbe7a06c/local_harbor_agents/patched_claude_code.py#L172-L187)

コピーしない具体例:

- SDK互換のhistory/schema/multimodal変換がない点を、そのままV2の機能省略につなげない。
- 「ReadツールでPNG/PDFを見る」というpromptだけを、VLM入力の代替にしない。[V3 reviewer prompt](https://github.com/findalexli/ai-scientist-v3/blob/b03021d78215271e5a5b2a894db6d3fadbe7a06c/.claude/agents/reviewer.md#L41-L48)
- ホストの認証ファイルをDocker build context/imageへコピーする方式を採用しない。[認証ファイルのコピー](https://github.com/findalexli/ai-scientist-v3/blob/b03021d78215271e5a5b2a894db6d3fadbe7a06c/run.sh#L323-L333)
- stderrとJSON stdoutの混合、`|| true`による終了コードの無視、非空ファイルだけの成功判定を採用しない。[reviewer実装](https://github.com/findalexli/ai-scientist-v3/blob/b03021d78215271e5a5b2a894db6d3fadbe7a06c/scripts/submit_for_review.sh#L163-L282)
- APIキーの有無によるCodex利用可否判定は、OAuthだけの利用を取りこぼすため採用しない。[ensemble backend選択](https://github.com/findalexli/ai-scientist-v3/blob/b03021d78215271e5a5b2a894db6d3fadbe7a06c/scripts/submit_for_review.sh#L104-L126)
- V3のREADMEと実コードにはreviewer/timeout既定値の差がある。記載値をそのまま本製品の仕様にしない。

## 5. 目標アーキテクチャ

```text
V2の全caller（探索worker・summary thread・執筆・査読）
  ↓ 既存互換facade: llm.py / vlm.py / treesearch.backend.query
共通InferenceGateway
  ├ role/backend解決・履歴正規化・画像・schema・意味検証
  ├ API transport（明示APIモードのみ。既存互換）
  └ CLI broker（run内で共通、全process/threadから接続）
       ├ CodexCliTransport → 利用者が認証した公式codex
       └ ClaudeCliTransport → 利用者が認証した公式claude
  ↓ 正規化結果 + usage/error → 既存facadeの戻り値へ変換
```

CLIは研究workspaceでコードを書かず、実験を起動しない。生成Python、metric抽出、plot、LaTeXコンパイルは、従来どおりV2側が実行する。CLIのsandboxを設定しても、V2の`Interpreter`やLaTeX等が自動的にsandbox化されるわけではない。

### 5.1 新設するモジュール案

`ai_scientist/inference/`を新設する。以下は責務単位であり、不要に細分化した実装を要求するものではない。

| モジュール案 | 責務 |
|---|---|
| `contracts.py` | Request/Result、Message/ContentPart、Usage、Capability、typed error |
| `config.py` | backend/role/model解決、CLI版と能力の検査、設定の直列化 |
| `gateway.py` | 共通呼び出し、batch展開、facade接続、結果検証 |
| `history.py` / `images.py` | caller指定historyの正規化、画像assetと掲載occurrenceの管理 |
| `schema.py` | FunctionSpec変換、JSON Schemaと意味検証、限定的repair |
| `broker.py` | run共通queue、provider別上限、request所有権、cancel |
| `subprocess_runner.py` | argv/stdin、並行stream読取、timeout、process group終了、ログ分離 |
| `codex_cli.py` / `claude_cli.py` | 各CLIのコマンド構築、入力形式、イベント解析、失敗の正規化 |
| `api.py` | 既存SDK経路を維持するtransport。CLIモードから到達不可 |
| `usage.py` / `artifacts.py` | usageの集約、request/run manifest、推論なしの停止時保存 |

API SDKを模倣した`.chat.completions.create`実装を新設することは推奨しない。既存公開helperの外形を残し、SDKの生responseを使用する処理はtransportの内側に閉じ込める。

### 5.2 共通契約

```text
InferenceRequest
  request_id, run_id, role, node_id/stage_id/ensemble_index（存在するとき）
  backend, requested_model（nullならCLI既定を明示利用）
  system_instruction, ordered_messages: Message[]
  output_mode: text | json_schema, original_schema?, semantic_validator?
  requested_parameters: temperature?, max_output_tokens?, ...
  queue_deadline, execution_deadline, retry_budget

ContentPart = TextPart | ImageOccurrence
ImageOccurrence = occurrence_id, asset_id, label, original_position
ImageAsset = actual_mime, byte_size, sha256, local_readonly_path

InferenceResult
  request_id, text?, structured_data?, completion_status
  requested_model, actual_model?, backend, cli_version?
  requested_parameters, effective_parameters, capability_warnings
  usage（取得できない値はnull）, duration, attempt_count
  artifact_refs, terminal_event, normalized_error?
```

既存prompt辞書のMarkdown変換は`treesearch/backend/utils.py:compile_prompt_to_md`を尊重する。textと画像を含むcontent listを単純に`str(list)`へ変換しない。

### 5.3 既存互換facadeの契約

| 関数 | 維持する戻り値・挙動 |
|---|---|
| `llm.create_client` / `vlm.create_client` | `(client_handle, model)`。CLIではSDKを生成しない |
| `treesearch.backend.get_ai_client` | 共通client handle。text helperへ渡しても利用可能 |
| 単発text/VLM helper | `(answer_text, updated_history)` |
| batch text/VLM helper | `(answer_texts, updated_histories)`。入力と出力の順序を保持 |
| `treesearch.backend.query` | 通常は`str`、FunctionSpec付きは検証済み`dict` |

clientのSDKクラス自体の同一性を保証する仕様ではない。repository内でSDK属性へ直接アクセスする箇所は移行し、外部callerに対しては互換変更として記録する。APIモードで既存SDK clientを引数に渡す用途は、必要に応じてAPI専用adapterで受ける。

## 6. 設定・モデル・認証

### 6.1 backendとmodelの分離

新設するbackend識別子は`api`、`codex_cli`、`claude_cli`とする。APIのprovider詳細は既存設定との互換を維持する。`claude`というmodel名だからBedrock、といった判定をCLIモードに持ち込まない。

モデルIDの一律置換は行わない。旧`gpt-4o`、`bedrock/...`等を現在のCodex/Claudeモデルへ黙って変換しない。CLIの既定モデルを使う場合は`model: null`としてモデル引数自体を省略し、選択結果が取得できる場合は記録する。再現性が必要なrunでは、利用可能性を確認した具体的なモデルIDを設定する。

統一roleは最低限以下を用意する:

`ideation`, `experiment_code`, `execution_feedback`, `experiment_vision`, `stage_control`, `node_selection`, `summary`, `report`, `aggregation`, `citation`, `writing`, `writing_vision`, `review_text`, `review_vision`。

一つの既存用途を不用意に別モデルへ変更しない。roleを細分化しても、既定の継承先と旧設定の対応を固定する。

設定案（**未実装の新仕様**。秒数は調整可能な初期値案）:

```yaml
inference:
  mode: cli
  subscription_only: true
  default:
    backend: codex_cli
    model: null
  roles: {}                 # 未指定roleはdefaultを継承
  providers:
    codex_cli:
      executable: codex
      max_in_flight: 1
    claude_cli:
      executable: claude
      max_in_flight: 1
  history_mode: explicit_stateless
  unsupported_parameter_policy: warn_and_record
  overflow_policy: error
  queue_timeout_seconds: 3600
  request_timeout_seconds: 1800
  transient_retry_limit: 2
  schema_repair_limit: 1
  fallback: none
```

Claudeだけで実行する場合は`default.backend: claude_cli`とする。役割別併用は、例えば`roles.writing.backend: claude_cli`で明示する。未導入のproviderは、どのroleからも参照されないなら必須にしない。

### 6.2 設定解決の優先順位と伝播

1. 明示されたCLI引数によるrole上書き。
2. inference設定ファイルのrole定義。
3. inference.defaultの継承。
4. **APIモードに限り**旧YAML/旧引数の既定値を互換解決。

新設引数案は`--inference-config`、`--llm-backend`、`--llm-model`。既存`--model_writeup`等は明示入力されたときだけ対象roleへ適用する。argparseの既定値とユーザー指定を区別しないと、CLI既定を旧APIモデルが上書きするため、値のprovenanceも管理する。

CLIモードでは`AVAILABLE_LLMS`/`AVAILABLE_VLMS`を共通のmodel allowlistにしない。providerの能力、選択モデルの利用可否、required capabilityで検証する。`subscription_only: true`とAPI roleの組み合わせは起動時エラーとする。

設定はlauncherからrun YAML、experiment driver、AgentManager、workerの`cfg`、summary thread、plotting、両writeup、最終reviewへ伝える。単独起動のideation/writeup/plottingにも同じ解決規則を使用する。親プロセスのglobal変数だけに依存せず、解決済み設定とbroker endpointを直列化して渡す。

### 6.3 サブスク認証を意図的に選ぶ

利用者が公式CLIで事前ログインし、そのCLIに認証管理を任せる。実装はブラウザ認証、token収集、`auth.json`の読み出し・複製・イメージ埋め込みを行わない。`HOME`、`CODEX_HOME`、`CLAUDE_CONFIG_DIR`を一時ディレクトリへ付け替えて既存認証を壊さない。

CodexはChatGPTログインとAPIキーで認証・課金経路が異なる。CLI adapterではChatGPT認証を要求し、対応版で`forced_login_method="chatgpt"`を用いる方針とする。[Codex認証](https://learn.chatgpt.com/docs/auth)、[設定リファレンス](https://learn.chatgpt.com/docs/config-file/config-reference)

Claudeの非対話モードでは、継承した`ANTHROPIC_API_KEY`がサブスク認証より優先される。cloud provider、別auth token、apiKeyHelper等も認証元へ影響するため、保存ログインだけを見てサブスク利用と断定しない。[認証の優先順位](https://code.claude.com/docs/en/authentication#authentication-precedence)、[環境変数](https://code.claude.com/docs/en/env-vars)

採用仕様:

- CLI子プロセス用の環境辞書を別に作り、APIキー、APIエンドポイント上書き、Bedrock/Vertex等のprovider切替を排除する。親環境やユーザーの設定ファイルは変更しない。
- `CODEX_API_KEY`、`OPENAI_API_KEY`、`ANTHROPIC_API_KEY`、`ANTHROPIC_AUTH_TOKEN`等の存在は、値を出力せずに検査する。具体的な除外対象は検証したCLI版ごとに定義する。
- Claudeの公式subscription tokenを利用する運用は、利用者が公式手順で設定する場合だけを対象とする。token生成や平文出力をアプリで自動化しない。
- auth status、設定、実行イベントから期待する認証元を確認できない場合は、認証設定不明として停止する。APIへの「とりあえず接続」は行わない。
- provider側の追加usage購入や契約設定までCLI wrapperで完全に制御できるとは主張しない。必要な課金上限はアカウント側でも設定する。

認証元の判定は**同じcwd・環境・settings・managed policyを適用した状態で、最初の推論送信より前**に行う。Claudeのsettings内の`env`は継承環境を上書きでき、active profile/gateway等も経路に影響するため、子プロセス環境からキーを除くだけでは十分ではない。`auth status`が今回のprint-mode設定を反映するかも固定版で確認する。実行後のusage/eventによる確認だけでは、初回の意図しない課金を防げない。[環境変数の優先順位](https://code.claude.com/docs/en/env-vars#precedence)

この設計の対象は利用者自身が操作する、未改変の公式CLIである。第三者のClaude資格情報を集めて代理実行するサービスや、独自のclaude.aiログイン実装とは区別する。配布・ホスティング形態を変える場合は、その時点の公式条件を再確認する。[Claude Codeの認証・利用条件](https://code.claude.com/docs/en/legal-and-compliance)

## 7. CLI transportの詳細

### 7.1 共通の実行契約

- `subprocess.Popen`等へ**argv配列**を渡し、`shell=False`とする。モデル出力、モデル名、ファイル名をshell文字列へ埋め込まない。
- リクエストごとに専用のprivateディレクトリを作る。CLIのcwdは研究repositoryではなくそのディレクトリとし、必要な画像・system文・schemaのみを置く。
- 長いprompt/historyはstdin、system文とCodex schemaはファイルを使う。論文本文、コード、画像base64をコマンドラインへ入れない。
- stdoutとstderrは分離して継続的に読み取る。片方のpipeが詰まって停止しない実装にする。
- CLIのstreamはtransportで解析し、既存callerへは最終回答だけを返す。progress、thinking、tool event、stderrを論文やJSONへ混入させない。
- 終了コード、terminal event、回答の存在、schema/意味検証を組み合わせて成功を判定する。「回答ファイルが非空」だけでは成功にしない。
- 対応版・モデル・能力を事前検査する。未確認のflagを黙って削って実行するcompatibility fallbackは行わない。

### 7.2 Codex CLI

基本方式は`codex exec`。`--json`はJSONLイベント、`--output-last-message`は最終回答の保存先、`--output-schema`は回答schemaの指定であり、それぞれ別の機能である。`--ephemeral`でCLI側のsession保存を抑制するが、これは自動context compactionの無効化ではない。[非対話実行](https://learn.chatgpt.com/docs/non-interactive-mode)

コマンド構築の骨格は次のとおり。これは**実装時にargvとして組み立てる設計図**であり、隔離設定の検証前にそのまま本番実行する完成コマンドではない。

```text
codex exec
  --json
  --ephemeral
  --ignore-user-config
  --skip-git-repo-check
  --sandbox read-only
  --color never
  --cd <request_directory>
  -c forced_login_method="chatgpt"
  -c approval_policy="never"
  -c model_instructions_file=<TOMLとして正しくencodeしたsystem_fileパス>
  --output-last-message <request_directory/final.txt>
  [--model <explicit_model>]
  [--image <image_1> --image <image_2> ...]
  [--output-schema <schema_file>]
  -
```

`--ignore-user-config`はユーザー設定の読込を抑えるが、認証情報の場所まで変更するものではない。画像は`--image`で実ファイルを渡す。上記の主要flagはローカル0.157.1のhelpとも照合した。[CLIリファレンス](https://learn.chatgpt.com/docs/developer-commands?surface=cli)

実装時に固定版で検証する隔離設定:

- repositoryの`AGENTS.md`やユーザーの研究用ルールが推論に混ざらないようにする。`project_doc_max_bytes=0`等の適用範囲を実測する。
- shell/unified_exec、hooks、multi-agent、memories、apps、web search等を不要なものから無効化する。候補は`features.shell_tool`、`features.unified_exec`、`features.hooks`、`features.multi_agent`、`features.memories`、`apps._default.enabled`、`web_search`。
- MCP、plugins、managed policyも確認する。`read-only`だけで全ツールが無効になるとは扱わない。
- CLIの画像添付に必要な内部読込は許容するが、モデル判断による研究コード実行・workspace変更・外部ツール利用は許可しない。

設定キーの根拠は[公式config reference](https://learn.chatgpt.com/docs/config-file/config-reference)。全ての組み合わせで「純粋な推論用profile」として動くかは未検証であり、Phase 0のリリースゲートとする。イベントから禁止ツール使用を検出する検査は追加防御であって、実行前の制限の代わりにはならない。

parserは少なくとも以下を扱う:

1. `thread.started`等からrequest追跡情報を取得。
2. `item.*`の通常assistant回答と他種イベントを区別。
3. `turn.completed`と終了コードを確認し、最終回答ファイルと整合させる。
4. `turn.failed`、`error`、terminal eventなしの終了を失敗として分類。
5. usageが取得できれば正規化する。cached inputをinputへ重複加算しない。

イベントschemaは版ごとのfixtureで管理する。進捗表示等の意味に影響しないものだけを読み飛ばし、未知の終了状態や、履歴・入力の変更に関わる可能性のある未知イベントはprotocol不明として扱う。compactionを検知できないことを「圧縮なし」の証明にしない。

### 7.3 Claude Code CLI

基本方式は`claude -p`。textだけの実装にはせず、画像と実行メタデータを扱える`stream-json`入出力を共通経路にする。

重要な禁止事項:

- **`--bare`を採用しない。** 現行仕様では保存されたサブスクOAuth/Keychain等を利用せず、今回の認証要件と合わない。[programmatic CLI](https://code.claude.com/docs/en/headless)
- 存在を確認していない`--input-format json`を使わない。入力はtextまたはstream-json。
- `--allowedTools`を「ツールを除去する指定」と誤解しない。
- `--dangerously-skip-permissions`、`bypassPermissions`を前提にしない。

候補profileのargv設計図:

```text
claude -p
  --input-format stream-json
  --output-format stream-json
  --verbose
  --no-session-persistence
  --system-prompt-file <system_file>
  --safe-mode
  --tools <空文字列のargv要素>
  --disallowedTools mcp__*
  --strict-mcp-config
  --mcp-config <empty_mcp_config_file>
  --settings <hooks_disabled_settings_file>
  --permission-prompts none
  [--restricted: 固定版で全オプションとの組み合わせを検証して採用]
  [--model <explicit_model>]
  [--json-schema <compact_schema_json>]
```

MCP設定の内容は`{"mcpServers":{}}`、settingsの追加内容は`{"disableAllHooks":true}`を基本とする。ただしsettingsはmergeされ、managed policyは別に作用する。空settingsだけで全環境設定が消えるとは扱わない。[settings precedence](https://code.claude.com/docs/en/settings#settings-precedence)、[hooks](https://code.claude.com/docs/en/hooks#disable-or-remove-hooks)、[MCP設定](https://code.claude.com/docs/en/mcp#project-scope)

操作ツールを除去してもhook自身はshell実行や追加入力を発生させ得る。一般操作・追加推論・入力変更を行う強制hook/helperが残り、排除を確認できない環境は、純粋推論profileとして未対応とし**推論前に停止**する。managed policyを無視・回避する実装にはしない。

`--safe-mode`は通常認証を維持しながらcustomizationを抑制する候補である。`--tools ""`はbuilt-in toolsを除去するがMCPは別管理となる。`--permission-prompts none`は現行ドキュメントで2.1.259以降。利用版のhelpだけでは全flagの有無を判定できない場合もあるため、公式仕様と実際のprotocol testを組み合わせる。[CLIリファレンス](https://code.claude.com/docs/en/cli-reference)

`--safe-mode`の導入は2.1.169、`--restricted`は2.1.248で確認できるが、これは全併用の動作保証版を意味しない。[固定CHANGELOG](https://github.com/anthropics/claude-code/blob/7779afb12e3635f46f56ec823979d68350ae000b/CHANGELOG.md#L3347)、[restricted導入箇所](https://github.com/anthropics/claude-code/blob/7779afb12e3635f46f56ec823979d68350ae000b/CHANGELOG.md#L1661)

特に**`--tools ""`と`--json-schema`の併用で内部の構造化出力処理が成立するかは未確認**。操作ツールを有効化しない状態で、text/画像の両方からschema準拠の回答が返ることをPhase 0で確認する。必要な内部schema処理と、shell/Read/MCP等の一般操作権限は分けて扱う。

入力は一要求につき一つのuser envelopeをNDJSONの一行として送る。以下は画像入力の概念例であり、`data`へ実際のbytesをbase64化して入れる。

```json
{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Figure occurrence fig-1. 指定された基準で評価してください。"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"<BASE64_BYTES>"}}]},"parent_tool_use_id":null}
```

この形式の根拠は[公式の画像付きstreaming input例](https://code.claude.com/docs/en/agent-sdk/streaming-vs-single-mode#implementation-example)と、公式Python SDKが[CLIをstream-jsonで起動する実装](https://github.com/anthropics/claude-agent-sdk-python/blob/36f95486ee9fc49d8ee1ed56811f07b5e8e23ac6/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L574)、[入力形式の指定](https://github.com/anthropics/claude-agent-sdk-python/blob/36f95486ee9fc49d8ee1ed56811f07b5e8e23ac6/src/claude_agent_sdk/_internal/transport/subprocess_cli.py#L793)、[NDJSONの送信](https://github.com/anthropics/claude-agent-sdk-python/blob/36f95486ee9fc49d8ee1ed56811f07b5e8e23ac6/src/claude_agent_sdk/_internal/query.py#L1112)である。SDKを依存に加える提案ではなく、公式CLIに到達するwire formatの根拠として参照している。今回standalone CLIでの実推論は行っていない。

parserは最終`result` envelopeの`subtype`、`is_error`、`stop_reason`、終了コードを検査する。textは`result`、schema要求では`structured_output`を取得する。成功subtypeでも要求したstructured dataがなければ成功にしない。stream途中のassistant/tool/thinkingを最終回答と誤認しない。[result型](https://code.claude.com/docs/en/agent-sdk/typescript#sdkresultmessage)、[structured outputs](https://code.claude.com/docs/en/agent-sdk/structured-outputs)

### 7.4 APIパラメータとの差

| 既存の要求 | CLIでの扱い |
|---|---|
| `temperature` | 同等の公開制御が確認できないproviderでは、要求値・非適用を記録して警告。再現性重視のstrict設定では事前エラー |
| `max_tokens` / `max_output_tokens` | CLIで同等制御を確認した範囲だけ適用。Claudeの`CLAUDE_CODE_MAX_OUTPUT_TOKENS`はモデル上限等があり完全互換ではない。Codexへ未確認flagを捏造しない |
| `n` | n個の独立した要求に展開。batch内の序数とhistoryを分離 |
| forced function/tool call | 任意の操作ツール実行ではなく、回答schemaとhost検証へ変換 |
| `system` / `assistant` role | system instructionと明示history表現へ変換。ただしAPIと同じrole権限を再現できるとは保証しない |
| API料金・token | providerが報告したusageを正規化。欠損はnullであり0ではない |

Claudeの出力上限・structured retry関連の環境変数は[公式env-vars](https://code.claude.com/docs/en/env-vars)を参照する。CLI内部retryとhost側repairが掛け算で膨らまないよう、対応版で設定できる範囲を確認して総試行数を制限する。`--max-turns 1`を無条件に付けると内部schema処理を妨げ得るため、単発推論の意味とCLI内部turn数を混同しない。

## 8. 会話履歴・画像・構造化応答

### 8.1 会話履歴: caller指定の範囲を唯一の正とする

初期実装は**stateless explicit replay**とする。各呼び出しで新しいCLIプロセス/会話を使い、渡されたhistoryと今回promptだけを、順序・roleラベル・content境界付きで一つの推論入力へまとめる。system instructionは別ファイルとして指定する。

重要な互換条件:

- `msg_history=None`、空配列、全履歴、`msg_history[-1:]`を区別する。ICBINBが意図的に残した最後の一件だけの履歴を、CLI側の過去sessionで補わない。
- 同一履歴からのensemble n件はそれぞれdeep-copyした独立branchとする。
- callerが渡した履歴をin-place変更しない。helperは従来形式の履歴に今回user/assistantを追加して返す。
- 独自のhistory表現はバージョンを持ち、文字列だけでなくtext/image partsを保持する。role区切り文字が本文に含まれても境界が壊れないescape方式にする。
- 長いhistoryを自動要約・切り捨てしない。context超過は明示的エラーとする。

adapterだけでなく、CLI内部の自動compactionも対象とする。Claudeは`DISABLE_AUTO_COMPACT=1` / `DISABLE_COMPACT=1`を候補として固定版で効果を確認する。Codexも内部圧縮の制御・検知をPhase 0で確認し、設定値の`0`を根拠なく「無効化」と解釈しない。入力削減なしで処理できることが確認できないサイズはpreflightで拒否し、未確認のまま完全な履歴保持を保証しない。[Claude環境変数](https://code.claude.com/docs/en/env-vars)

Codexの`model_auto_compact_token_limit`は発火しきい値であり、無効化用の特殊値は今回確認できていない。停止手段の候補として、公式には`PreCompact` hookが`{"continue":false}`を返すと圧縮前に停止する仕組みがある。採用する場合は7.2節のhook無効化を「不要なhookを無効化し、host管理の停止専用hookだけを許可」と調整し、trust、実動作、停止時のexit/event、hook失敗時を固定版で検証する。これは採用確定ではない。またapp-serverのcompaction eventが`exec --json`にも同じ形式で出るとは仮定しない。[Codex設定](https://learn.chatgpt.com/docs/config-file/config-reference)、[PreCompact](https://learn.chatgpt.com/docs/hooks#precompact)、[app-serverのitems](https://learn.chatgpt.com/docs/app-server#items)

Claudeのstreaming inputに過去の各messageを別々のuser turnとして流す実装は採用しない。それは履歴の注入ではなく、追加の生成turnを発生させるためである。native resume/continueはsessionを共有・蓄積するので、nodeやreflectionごとの明示historyを尊重するには別設計が必要になる。[session semantics](https://code.claude.com/docs/en/agent-sdk/sessions)

制約として、role付き履歴を一つのuser contentへ描画する方式はAPI message列の厳密な再現ではない。内容・順序・分岐は維持するが、モデル内のrole優先度は異なる。この差を受入試験で確認し、必要になった場合だけ、Codex app-server等のnative履歴入力を別transportとして検討する。最初からnative sessionを共有して高速化しない。[Codex app-server](https://learn.chatgpt.com/docs/app-server)

### 8.2 画像: bytesと掲載位置を分けて管理

全VLM経路で、画像パスの文字列を見せるだけではなく画像bytesを送る。Codexは正規化した実ファイルを`--image`へ、Claudeはbase64 image partへ変換する。

採用仕様:

1. 既存path/data URI/content blockを共通ImageAssetへ解決し、存在、読み取り可否、実形式、サイズを確認する。
2. MIMEは実際のbytesと一致させる。PNGを変換せず`image/jpeg`と宣言しない。変換した場合は変換後の形式・hash・寸法も記録する。
3. `asset_id`はbytes共有のため、`occurrence_id`は同じ図の別掲載位置を区別するために使用する。hashが同じでもoccurrenceを削除しない。
4. 画像順序、対応label、元ファイル、history内位置をmanifestへ記録する。Codexの初期画像添付のように平坦化する場合も、promptに対応表を付ける。
5. モデルから返された図ID・件数・順序を、実際に送信したoccurrence集合と照合する。既存schemaに図IDがない場合の方法は次の段落で明示する。
6. 外部URLが入力に現れる場合は、許可した取得処理でbytesに解決するか未対応として停止する。CLIへURLだけを渡して「画像を送った」と扱わない。

現行`vlm_feedback_spec`の`plot_analyses`要素は`analysis`しか要求しておらず、返却図IDを検証できない。この経路では内部生成schemaに対応確認専用の`occurrence_id`を追加し、提出したID集合・件数をhostで検証してから、実際の送信順へ並べ直して既存形式へ投影する。件数の要求は8.3節の否定評価ルールに従い、正当な否定フラグ＋空解析には全件回答を要求しない。元の科学的評価fieldは変更しない。`MinimalAgent`側も元の`node.plot_paths[index]`ではなく、送信した`selected_plots`と検証済み対応表を使用する。これは管理metadataの明示的な拡張であり、モデルが返していないIDを後から正解として補う処理ではない。

V2維持の対象:

| 画像用途 | 維持する評価・影響 |
|---|---|
| 探索plotの選択 | 現行の多数plotからの選択工程と`selected_plots`を維持 |
| 探索plot解析 | `plot_analyses`、`valid_plots_received`、`vlm_feedback_summary`。good node/段階判定へ接続 |
| 執筆前の図説明 | `Img_description`、`Img_review` |
| caption/本文参照レビュー | `Caption_review`、`Figrefs_review`等 |
| 掲載・配置・情報量のレビュー | `Overall_comments`、`Containing_sub_figures`、`Informative_review`等 |
| 重複図検出 | 同一要求内の複数図比較。別掲載位置も維持 |
| 最終図レビュー | text査読とは別に実行し、既存のreview成果物へ保存 |

既存single VLMの25枚、batchの200枚への暗黙sliceは、CLI対応を理由に踏襲しない。上限超過はエラーとし、必要なら全件coverageを保つ分割方式を明示設計する。特に重複図検出ではグループ内比較だけではグループ間重複を見逃すため、単純分割は互換ではない。

CLIや選択モデルの画像枚数・bytes・解像度上限は実測する。Claude APIの画像制限値をそのままsubscription CLIへ流用しない。[Visionのinterface別制約](https://platform.claude.com/docs/en/build-with-claude/vision#request-limits)

### 8.3 構造化応答と意味検証

`FunctionSpec`はモデルに要求する結果の契約であり、CLIへ任意関数の実行権限を与えるためのものではない。現行のauthoritative schemaは`parallel_agent.py:81/103/135/205`、`journal.py:23`、`agent_manager.py:21/49/78/594`等にある。

処理順序:

1. 元のschemaを取り込み、providerのnative schema機能へ変換する。
2. providerが許すJSON Schemaのsubsetを確認する。変換する場合も元schemaの意味を保持し、独自のrequired/default値追加で回答を捏造しない。
3. terminal responseからJSONを取り出す。
4. 元schemaに対するhost側の検証を必ず行う。
5. schemaだけでは保証できない意味制約を検証する。
6. 内容不正の場合だけ、許可された回数のrepairを行う。修正要求と追加コストを記録し、なお不正ならtyped errorを返す。

providerがnative schemaに対応しない場合、明示されたJSON生成＋host検証方式で同等の出力契約を満たせるかを別途評価する。未検証の方式へ無言でfallbackして「structured対応」と宣言しない。

意味検証には、参照node IDが候補集合にあること、plot IDが提出画像に対応すること、要求件数、重複回答・欠落、有限な数値、既存スコア範囲、Accept/Rejectの値、必要なfileの存在等を含める。空配列がschemaを通ることだけを成功根拠にしない。

画像を送信する前の選択結果も検証する。`selected_plots`は元の候補`node.plot_paths`の部分集合でなければならず、「実在する画像ファイル」であることだけでは不十分である。

ただし**正常な否定的評価を不正応答と混同しない**。`valid_metrics_received=false`や`valid_plots_received=false`は正式な評価結果であり、その否定フラグと既存契約に整合する空配列は正常に返す。肯定フラグなのに必須の解析・対応が欠ける場合と区別し、悪い実験結果を良い結果へ変えるためのrepairは行わない。transport成功と実験成功は別の状態である。

一方、コード・THOUGHT・fenced JSON・LaTeXを含む既存text出力へ一律にJSON schemaを強制しない。textモードは既存parserとの互換を維持し、structuredモードが必要なcallerだけdictを受け取る。途中打切りのJSON/LaTeXを完成成果物として採用しない。

### 8.4 査読ensembleの意味を維持

- n件を独立生成し、返却順序と各historyを保存する。同一回答の複製、session共有、同一cache結果の使い回しで件数を満たさない。
- 現行の16項目のreview契約を維持する。7つの評価値は1〜4、Overallは1〜10、Confidenceは1〜5、倫理フラグはbool、DecisionはAccept/Reject。
- meta-reviewとreflectionを残す。集約対象の数値9項目は現行どおり有効値の平均を`round`で整数化し、Decisionを独自の多数決へ変更しない。
- 有効回答数、失敗数、要求数を明示し、不足を無言で成功扱いしない。既存の有効回答処理と、transport自体の失敗を区別する。
- 論文全文やfew-shot例をstdinで扱い、argv制限回避のために入力を削らない。stdin自体にも上限があり得るため、大きなtextと画像envelopeの双方を試験する。

## 9. 並列実行・timeout・プロセス所有権

### 9.1 run共通brokerを置く理由

探索は`ProcessPoolExecutor`、最終要約は`ThreadPoolExecutor`を使い、査読batchも複数要求を発生させる。各Pythonプロセスが個別のsemaphoreを持つだけでは、合計CLI数を制限できない。

初期実装では**runごとに一つのbroker process**を置き、CLIの起動をそこへ集約する。親、探索worker、summary thread、査読batchは同じbrokerへ要求する。APIモードをこの変更のためだけに全面的なIPC化する必要はない。

採用仕様:

- 初期の対応環境は既存実験環境に合わせたLinux/macOS。IPCは権限を制限したUnix domain socket等を使用し、短いprivate pathを割り当てる。Windows対応は別検証とする。
- broker endpointと解決済み設定は直列化可能にし、worker初期化時に接続する。Popen、lock、session、open済みsocketをNodeやcfgへpickleしない。
- request/responseは明示schemaを持つ。汎用の外部pickle読込や外部公開HTTP proxyにはしない。
- providerごとに`max_in_flight=1`から開始する。GPU worker数とは独立させる。両CLIを使う場合も各枠を明示する。
- 有限queue、fair scheduling、backpressure、queue timeoutを設ける。大量のensembleが他roleを無期限に占有しないようにする。
- request_idを採番し、batch、node、reflectionを区別する。正常完了した同一requestへの配送重複は防いでも、独立ensemble要求を同一promptとしてcache統合しない。
- CLIの完了順にかかわらず、探索Nodeは元のfuture投入順でJournalへ登録する。通常時の次バッチの親選択も現行のバッチ境界を維持し、完了順appendや部分結果だけでの次バッチ起動へ変更しない。`Journal.append()`による`node.step`と要約順序を保存するためである。
- broker異常終了は全接続callerへtransport failureとして通知する。接続断だけを根拠に「未実行」と判断して自動再送しない。

この上限が保証するのは**一つのrun内の並列数**であり、別runやユーザーが手動起動したCLIまで含むアカウント全体の上限ではない。同時に複数runを動かす運用には、別途共通broker/lockまたは運用上の制限が必要となる。

### 9.2 四種類の時間を分離する

| 時間 | 起点・対象 | 超過時 |
|---|---|---|
| queue待機時間 | broker受理からCLI起動前まで | 未実行のままqueue timeout |
| CLI推論時間 | CLIプロセスの起動から回収完了まで | 所有するCLI process groupを停止し、typed timeout |
| 実験時間 | `Interpreter`による生成コードの実行 | 従来の実験timeout。ただし確実なcleanupを必要とする |
| job/run期限 | 上記待機・実験・post-processingを含む全体 | cancelを伝播し、停止状態を保存 |

現在`cfg.exec.timeout`がfuture待機にも使われる箇所があるため、同じ値へCLI待ち時間を押し込まない。実験のtimeoutを延長するだけではCLIの問題を解決できない。

### 9.3 終了とcancel

- POSIXではCLIを独立process groupとして起動し、通常終了、TERM、猶予後KILL、wait/reapの順に所有プロセスを処理する。
- stdout/stderr readerにも停止期限を設ける。強制終了後に存在しないEOFを無限に待たない。
- workerが停止・死亡したとき、broker上の対応requestをcancelする。親が終了する場合は未投入・queue中・実行中を区別する。
- `future.result(timeout=...)`だけではworkerを止められない。実験プロセスの停止確認前にGPUを解放・再割当しない。
- CLIが共有daemon等を使用する場合も、ユーザーの別sessionやdaemonを無差別にkillしない。固定版で所有範囲と終了挙動を確認する。
- `launch_scientist_bfts.py:347`のOS全体をキーワード検索してkillする終了処理は、runが所有するPID/process groupだけを処理する方式へ置換する。ユーザーのCodex/Claudeや別研究runを巻き込まない。

CLI側のツール制限と、V2が生成コードを実行する安全性は別問題である。既存Interpreterの危険性がCLI移行だけで解消したとは説明しない。

## 10. エラー・利用上限・成果物保存

### 10.1 typed errorの分類

| 分類 | 例 | 挙動 |
|---|---|---|
| `AuthenticationRequired` | 未ログイン、期限切れ、想定と異なる認証元 | runを明示停止。ログインを自動操作しない |
| `QuotaExhausted` | サブスク枠到達、追加利用が必要 | 新規要求を止め、成果物保存。API/別アカウントへ逃がさない |
| `UnsupportedCapability` | 対象版で画像/schema/隔離profile未対応 | 原則preflightで失敗。VLM等を省略しない |
| `ContextLimitExceeded` | prompt/画像/出力上限 | 入力を黙って削らず停止。分割等は明示設計のみ |
| `TransientProviderError` | 一時的通信断、短期的混雑 | 有限retryとbackoff。実行済みか不明な再送は別扱い |
| `QueueTimeout` / `InferenceTimeout` | 待機/実行期限超過 | request状態と部分ログを保存。科学的評価と分離 |
| `InvalidStructuredOutput` / `IncompleteOutput` | JSON不正、schema違反、途中切断 | bounded repairまたは明示失敗 |
| `Cancelled` / `BrokerUnavailable` | 利用者中断、broker障害 | 終了を伝播し、完了済み成果を保持 |

CLIの終了コードだけでquota/authを完全に判別できるとは限らない。固定版のイベント・stderr fixtureで認識し、不明なものはprovider errorとして残す。根拠のない「quotaらしい」判定で長時間待機しない。

### 10.2 科学的失敗へ変換しない

次のような既存の広い`except`は、infrastructure errorを先に再送出するように修正する。

| 箇所 | 現在起こり得る変換 |
|---|---|
| `parallel_agent.py:1654`付近 | metric抽出失敗を`WorstMetricValue` / buggy nodeに変換 |
| `parallel_agent.py:1766`以降 | plot/VLMの例外をlogに残して処理続行 |
| `parallel_agent.py:1327`付近 | multi-seed失敗を一般例外として処理 |
| `parallel_agent.py:2172`付近 | `TimeoutError`をfuture待機の問題としてlogだけで処理 |
| `journal.py:498`付近 | LLMによる選択失敗をmetric rankingへfallback |
| `agent_manager.py:388/496`付近 | 判定失敗をstage未完了として扱う |
| `log_summarization.py:214/287`付近 | 認証・quotaでも要約retryへ入る |
| `perform_icbinb_writeup.py:976`付近 | 図説明失敗を吸収する |

`valid_plots_received=false`という正常な評価結果と、画像を送信できなかった失敗は区別する。後者で`is_buggy_plots=True/False`を勝手に確定させず、評価未完了状態とtransport errorを記録する。研究用のdebug/retry予算をquotaエラーで消費しない。

typed inference errorは既存のfuture待機timeoutより先に捕捉・上位伝播する。特に新しい`QueueTimeout`/`InferenceTimeout`を組込み`TimeoutError`の派生にする場合、既存catchへ吸収されない順序を保証する。

### 10.3 推論を呼ばない保存経路

停止時に必要な保存処理はLLMから独立させる。現在のstep callbackは要約の後に保存へ進む経路があり、`save_run()`も最良node選択からLLMを呼び得るため、そのままではquota時の保存を保証できない。

最低限、次の順序を保証する:

1. 完了済みNode/Journal、設定、コード・実験結果・plotの参照を推論なしでsnapshotへ保存。
2. 正常運転時だけ、LLM要約や追加のbest-node選択を実施。
3. 成功した追加結果を保存。要約失敗でも1の成果を失わない。

親Journalだけの保存では不十分である。現在のworkerは一連の評価を終えてからNodeを返すため、実験成功後のmetric/VLMで停止すると、親にまだ渡っていないExecutionResultがある。workerは失敗時にも生成コード、取得済みExecutionResult、生成物参照、未完了phaseを推論なしで保存し、その参照を親へ伝える。親は新規投入を停止したうえで、既に完了した他futureの結果も回収・保存する。最初に失敗したfutureだけを見て回収ループ全体を打ち切らない。この保存契約は途中phaseからの自動resumeを要求するものではない。

run manifestには`completed`、`stopped_quota`、`stopped_auth`、`failed_transport`、`cancelled`等を残す。未完了node/評価は未完了と記録し、成功nodeに偽装しない。最低限のmanifest項目は次のとおり。

```text
schema_version, run_id, source_revision, config_hash
resolved_roles, cli_versions, verified_capabilities
run_status, last_stage, completed_node_ids, pending_evaluations
request_ids, attempts, normalized_errors, usage_summary
artifact_paths, completion_flags, stop_reason
```

auth token、全環境変数、資格情報ファイルは保存しない。raw promptや論文・画像も機密性を持ち得るため、private権限と保持方針を設定する。`--ephemeral`や`--no-session-persistence`を使っても、アプリ自身が保存したログまで消えるわけではない。

`perform_experiments_bfts_with_agentmanager.py:68/73`には、更新されない`global_step`に基づくatexit削除経路がある。中断・quota時のworkspace削除を避け、通常終了時の一時copy整理と永続成果物の保持を明確に分ける。

### 10.4 resumeは別の完成条件

現状にはstage終了付近のpickle保存があるが、完全な読込・途中復帰経路は確認できていない。`Journal.from_dict`の欠如等もあり、「既存resumeを使えばよい」とはしない。

- **初期CLI対応の必須範囲**: 正しく停止し、完了成果と未完了状態を保存する。再実行時に全てを自動再利用できるとは宣言しない。
- **後続の小規模拡張**: 明確なstage/batch境界から再開。どの範囲が再実行されるかを明示する。
- **さらに後の拡張**: 実験後のmetric/plot/VLM境界からの復帰、request cache、crash後in-flight照合、RNG、Journal graphの完全round-trip。

サブスク上限による中断は現実的に起き得るため、保存は初期対応の必須要件とする。一方、任意地点からの厳密なresumeをtransport置換の前提にして実装範囲を過大化しない。

## 11. 利用量・成果物・運用

### 11.1 token trackerの変更

`utils/token_tracker.py`のSDK response前提を、正規化Usageを受け取れる構成に拡張する。各CLIのresult/eventから取得し、brokerでrun全体を集約する。workerごとのglobal trackerだけでは全体を数えられない。

- input/output/cached input/reasoning等を区別し、providerごとの包含関係を記録する。
- 同一messageのevent重複や、累積usageの繰り返し加算を防ぐ。
- 値がない場合はnull/unknownとする。文字数からの推定を実測tokenとして表示しない。
- `subscription_billing_cost`は原則unknownとし、CLIが返す推定API相当costとは別欄にする。
- requested/effective sampling、実モデルが不明な場合のunknown、CLI版、schema repairの追加試行も残す。

Claudeの表示costはサブスク請求額そのものではなく、追加usageには別の課金が発生し得る。固定回数のV2 runが必ず契約枠内に収まるとも約束しない。[usageとcost](https://code.claude.com/docs/en/costs)

現在launcherのtoken snapshotは最終reviewより前にも出力される。**最終text/VLM reviewと終了処理の後**にrun全体の最終集計を残す。エラー停止でも最後に取得できたusageと未計測requestを表示する。

### 11.2 既存成果物の互換

既存のexperiment directory、生成コード、`.npy`結果、PNG、Journal/JSON要約、HTML可視化、LaTeX/BibTeX/PDF、text/画像reviewを残す。標準ICBINBの`*_reflection_final_page_limit.pdf`等の出力契約を、transport変更の都合で別形式へ置き換えない。

追加する診断情報は、例えば`<run>/inference/requests/<request_id>/`と`<run>/inference/manifest.json`へ分離する。schema、最終回答、stderr、正規化メタデータを保存し、raw streamは保持設定を設ける。画像・長いpromptは重複保存量に配慮しつつ、発生箇所と内容を追跡可能にする。

初期実装で恒久的なprompt cacheを自動有効化しない。特にensembleやtemperature依存処理をcacheで同一化するとV2の意味が変わる。

### 11.3 起動前診断

新設するdoctor/preflight（名称例: `python -m ai_scientist.inference.doctor`）は、通常診断と実推論probeを分ける。

通常診断:

- 解決された全role、backend、model、required capabilityを一覧化。
- 使用するCLIだけの存在・版・認証元・設定整合性を検査。
- API系credential/provider上書きの継承を、値を出さずに検知。
- image/schema/history/出力長の対応情報と、未検証の組み合わせを報告。
- CLI並列上限、timeout、ログ保存先、API fallback無効を表示。

実推論probeは利用枠を消費するため明示オプションとする。小さなtext、複数画像、schema、長文、隔離profileを確認し、モデル・版・日時付きの結果を保存する。過去のprobe成功を別モデル・別CLI版の保証には使わない。

V2のnode予算20/12/12/18等は推論回数ではない。debug、plot、評価、要約、reflectionで複数呼び出しになるため、run前の見積りは幅を持たせる。必要に応じてrequest数やrun時間の上限を追加するが、サブスク残枠を正確に取得できるAPIがあるとは仮定しない。

生成された研究コードが独自にLLM SDK/外部HTTPを利用する場合、それは本体gatewayの外側になる。`subscription_only`の保証範囲を本体管理の推論と明示し、生成コード用環境にLLM APIキーを渡さない。LLMを研究対象にする実験は共通gatewayを使う別の明示契約が必要であり、任意のPythonの全ネットワーク通信を自動変換できるとは説明しない。

## 12. ファイル別の変更計画

| ファイル・範囲 | 必要な変更 | 維持するもの |
|---|---|---|
| 新設`inference/` | 契約、設定、CLI transport、broker、検証、記録、doctor | V2から独立した小さな推論境界 |
| `llm.py` | CLI分岐、共通handle、単発/batch履歴、API経路の明示化、usage接続 | 既存関数名、text/history、APIモード |
| `vlm.py` | textと同じbackend解決、画像添付、batch、上限の明示化 | VLMの全public capability |
| `treesearch/backend/__init__.py` | model文字列dispatchから設定dispatchへ | `get_ai_client`と`query`のcaller契約 |
| `backend_openai.py` / `backend_anthropic.py` | API専用経路として整理し、共通result/usageへ | 明示API利用時の既存provider対応 |
| `treesearch/backend/utils.py` | FunctionSpecのhost応答検証、CLI用schema変換 | 元schema、promptのMarkdown化とmultimodal保持 |
| `perform_vlm_review.py` | 直接SDK呼び出し除去、画像ID/MIME、共通client | 重複・caption・参照・掲載の各評価 |
| `perform_llm_review.py` | batch独立性、schema/意味検証、typed error、usage | ensemble、meta-review、reflection、採点集約 |
| `perform_writeup.py` / `perform_icbinb_writeup.py` | runtime/roleを渡し、visionとtextを接続、失敗伝播 | 両形式の執筆・引用・図・ページ制限 |
| `perform_ideation_temp_free.py` / `perform_plotting.py` | standalone引数とconfig伝播、allowlistの適用範囲修正 | 既存の生成・修正処理 |
| `bfts_config.yaml` / `treesearch/utils/config.py` / `bfts_utils.py` | inference設定、解決済み設定のcopy、推論不要snapshot | 研究予算、実験設定、run directoryの配置 |
| `treesearch/parallel_agent.py` | broker接続、schema/image対応、typed error、job/GPU所有権 | 探索、multi-seed、metric/plot/VLMの流れ |
| `treesearch/agent_manager.py` / `journal.py` | 全role接続、選択/判定例外の分離、保存の非推論化 | 段階遷移とNode/Journalの科学的判定 |
| `log_summarization.py` / `journal2report.py` | hidden model既定値の解消、共通handle、threadからbroker利用 | stage要約、注釈、補助report機能 |
| `perform_experiments_bfts_with_agentmanager.py` | broker lifecycle、snapshot順序、中断時workspace保持 | 既存driverの研究管理 |
| `treesearch/interpreter.py` | cancel/終了期限の必要箇所を修正、所有プロセス管理 | V2による生成コード実行 |
| `launch_scientist_bfts.py` | 新引数、preflight、全工程へruntime伝播、限定的cleanup、最終usage | 起動から最終査読までの順序 |
| `utils/token_tracker.py` | provider-neutral usage、全process集約、unknown扱い | APIモードのtoken情報 |
| `requirements.txt` / README / 新規tests | CLI導入・認証・制約の説明、必要最小限のテスト依存 | API利用方法と既存実験依存 |

CLIモードではSDK constructorを起動しない。SDK importを遅延化すると依存分離しやすいが、既存API機能を削除してrequirementsからSDKを一括除去することは本作業の目的ではない。

## 13. 既存欠陥とCLI移行を混同しない

以下は静的読解で確認・抽出されたもので、今回の推論実行で再現したものではない。transport変更と分けたテスト・commitで扱う。

### 13.1 CLI対応の成立に直接必要な修正

| 問題 | 根拠箇所 | 修正の目的 |
|---|---|---|
| batch fallbackが渡されたhistoryを捨てる | `llm.py:187` | n回のCLI呼び出しで履歴を保持 |
| 一部batchのtoken decoratorがtupleとSDK responseを混同 | `llm.py`、`utils/token_tracker.py` | 正規化usageへ移して結果契約を壊さない |
| 選択画像の解析を元のplot indexへ対応付ける | `parallel_agent.py:1025` | 画像選択後のID対応を正す |
| PNG等のbytesをJPEG MIMEと宣言 | `parallel_agent.py:1001`、`perform_vlm_review.py:427` | native画像入力の正確性 |
| 画像数の暗黙slice | `vlm.py:156/289` | 未評価の図を黙って落とさない |
| CLI障害を一般研究失敗へ変換し得る | 10.2節の各`except` | quota/auth/cancelの伝播 |
| 保存がLLMに依存、atexitで成果削除の可能性 | driver、`utils/config.py:245` | 中断時にも成果を保持 |
| OS全体のkeyword kill、timeout後の所有権が曖昧 | launcher、parallel_agent、Interpreter | CLIや別研究runを巻き込まない |

### 13.2 フルパイプラインの結合試験で別途扱うもの

- 通常writeupの`msg=`指定とhelperの`prompt=`引数の不一致。
- node summaryが`FunctionSpec`を期待する箇所へdictを渡す経路（`parallel_agent.py:1063`）。
- substage進行とstage番号、四つ固定のsummary unpackの整合性。
- plot実行結果の保存先とretry判定が別fieldを見ている箇所、stage plot引数の順序。
- multi-seed submit時に同じdictを変更して渡す点、seed側のGPU releaseと固定件数を仮定した集約。
- review/page reflectionの未展開placeholder、reflectionがない場合の最終PDF選択。
- Interpreter強制終了後のEOF待ち、executor終了後のprivate属性依存。

これらを「CLIが悪い」と誤診しない。V2の全経路の受入試験を妨げるものは最小限の別修正として実施し、未解決の経路が残るなら「V2完全対応」と宣言しない。一方、全研究アルゴリズムの改善や一般的なrefactorまでは同時に広げない。

InterpreterのEOF待ちとexecutor cleanupは既存欠陥として別commitに分けても、Phase 4の「有限時間で終了する」ゲートには含める。CLI公開後へ無条件に先送りしない。

`Journal.from_dict`、serialization round-trip、任意地点resumeの整備は10.4節の後続拡張に分類する。ただし初期版の停止時snapshotは読込機能の有無にかかわらず保存できるようにする。

## 14. 実装順序と各段階の完了条件

| Phase | 実装・調査内容 | 次へ進む条件 |
|---|---|---|
| 0: CLI protocol検証 | 対応版を選定。利用者の明示実行で認証、text、複数画像、schema、長文、隔離profile、terminal errorを小さく検証 | 両CLIについて必要能力が確認でき、未対応を隠さず説明できる。特にClaudeのtools/schema併用とCodexの隔離を解決 |
| 1: 共通境界とfake CLI | contracts、config、runner、parser、schema、image/history変換、fake CLI fixture | APIキー・実CLIなしで契約テストと異常系テストが通る |
| 2: text/VLM facade | `llm.py`、`vlm.py`、direct duplicate call、usageを接続 | 単発・batch・複数画像・history slice・レビューが両transportで成立 |
| 3: 探索と全role | tree backend、設定伝播、broker、summary、stage/selection schemaを接続 | 全入口のCLI到達、SDK/LLM HTTP迂回路ゼロ、process/thread合計の並列上限を検証 |
| 4: 停止と保存 | typed error伝播、process ownership、GPU lifecycle、非推論snapshot | quota/auth/cancel時に科学的状態を偽装せず、成果を失わず有限時間で終了 |
| 5: V2全工程の回帰 | 既存欠陥を切り分け、両writeup・図・査読・artifactまで検証 | Codex-only/Claude-only/APIで受入表を満たす。違いを記録 |
| 6: 文書化・運用 | README、設定例、doctor、バージョン表、制限・usage説明 | 初期認証から停止時の確認方法まで利用者が再現できる |

各Phaseは既存APIモードを壊さない小さなcommitへ分割する。live probeや実験は契約枠・GPUを消費するため、仕様書作成だけの段階では実行しない。

最初の実装対象はCLIを呼ぶ短いwrapper単体ではなく、Phase 0で必要能力を確認した後の「共通契約＋fake CLIテスト」とする。これにより、サブスク利用枠を消費せず大半の接続・エラー処理を検証できる。

## 15. 受入試験

### 15.1 offline・決定的テスト

実際のLLMの回答一致を期待せず、固定fixtureを使ってV2側の意味を確認する。

| 試験 | 合格条件 |
|---|---|
| API迂回防止 | CLIモードでOpenAI/Anthropic/Bedrock等のconstructorと既知の直接LLM HTTP呼び出しを失敗させても、全roleがfake CLIで完走 |
| 入口網羅 | launcher、standalone ideation/plot/writeup、tree factoryからtext helperを呼ぶsummaryを含める |
| 結果解析 | progress、tool、thinking、stderr、複数イベント、非0終了、terminalなし、空回答、途中JSONを区別 |
| 履歴 | None/空/全履歴/最後の一件、text/image parts、並列branchで混入がない |
| batch | n個の独立request、入力順序、履歴独立、部分失敗と件数の明示 |
| schema | 不正JSON、欠落、型違い、未知ID、NaN等、無効画像対応を拒否。否定フラグ＋整合する空配列は正常応答でrepair 0回 |
| 画像 | 複数枚、選択後順序、同一bytesの別occurrence、正しいMIME、missing file、実在する候補外画像、上限超過を確認 |
| 長文 | argv上限を超える規模のpromptをstdinで扱い、内容を削らない。大きなNDJSON一行でもreader/writerが壊れない |
| 並列 | 複数process＋thread＋batchでもprovider上限以下。queue timeout、client死亡、broker死亡を検証。逆順完了でもNode登録順・次回選択を維持 |
| cleanup | CLI/実験の子を含むowned processを終了し、無関係なdummy processは生存。GPU解放は停止確認後 |
| quota/auth | 実験完了直後・metric前・VLM前・writeup途中・最終reviewで失敗注入。親に未返却の実験結果と、後続futureだけ先に完了した結果も保持 |
| usage | 未計測はunknown、cache/cumulative/event重複を二重計上せず、最終reviewまで合計 |
| API回帰 | 旧API設定の解決・引数・返却形式をrecorded/mock responseで検証 |

API迂回テストで遮断するのは本体のLLM transportであり、Semantic Scholar等まで「API」という名称だけで一括禁止しない。

### 15.2 V2の挙動を確認する固定fixture

1. `selected_plots → 送信画像 → plot_analyses → good_nodes → stage判定 → summary/HTML`の対応を確認する。
2. code/debug/improve、metric parse、baseline/creative/ablation、multi-seed集約を、小さな実験fixtureで通す。
3. `図説明 → caption/参照 → 重複判定 → 掲載判断 → ページ制限 → 最終図review`を両writeupで適用可能な経路ごとに確認する。
4. paper reviewは単独、ensemble、meta-review、reflectionと9数値の集約を確認する。
5. 通常writeupとICBINB双方でLaTeX/PDF、引用、図、review成果物の期待契約を照合する。
6. 同じ正規化fixture回答をAPI/CLI facadeへ供給したとき、V2側のNode選択・状態・成果物が同じになることを確認する。
7. 図のあるPDFからcaption/figrefs/画像occurrenceを抽出し、VLM要求とreview成果物まで照合する。抽出0件で無言終了する退行を検出し、本当に図のないPDFの正常な0件とは区別する。

これは新モデルが旧モデルと同じ研究結果を出すことを要求する試験ではなく、transportの違いだけでV2の判定や工程が変わらないことを確認する試験である。

### 15.3 live smokeと最終受入マトリクス

利用者が公式CLIへログインした環境で、明示的に実施する。

| 実行モード | text | 複数画像＋schema | tree/stage | 両writeup | text/VLM review | API自動fallbackなし |
|---|---|---|---|---|---|---|
| Codex CLIのみ | 必須 | 必須 | 必須 | 必須 | 必須 | 必須 |
| Claude Code CLIのみ | 必須 | 必須 | 必須 | 必須 | 必須 | 必須 |
| 役割別のCLI併用 | 必須 | 必須 | 必須 | 必須 | 必須 | 必須 |
| 既存APIモード | 互換回帰 | 既存対応範囲 | 互換回帰 | 互換回帰 | 互換回帰 | CLIへ勝手に変更しない |

最初は小さい実験設定で全経路を確認し、その後に通常のV2設定で性能・上限・長時間実行を評価する。smoke用に一時的に予算を減らすことと、製品の既定機能を削ることを混同しない。

CLIのtoolsを無効にしたprofileでもschemaと画像が成立すること、利用者のAGENTS/CLAUDE設定・hooks/MCP等が結果や副作用へ混入しないことを検証する。managed policyが隔離要件と両立しない環境は明示的に未対応とする。

### 15.4 完了の定義

- このrepositoryの本体管理の全推論経路が共通境界へ接続されている。
- Codex-only、Claude-onlyともにVLMを含むV2の必須工程を省略せず動く。
- APIへ無言でfallbackせず、認証元・backend・モデル差を表示する。
- 利用上限や中断で完了実験を失わず、研究上の失敗と実行基盤の失敗を区別する。
- 全工程の完了状態をmanifestで追跡でき、最終reviewを含む成果物とusageを保存する。
- 未解決の制限、対応CLI版、APIとの意味差、resume未提供の範囲がREADMEに明記されている。

## 16. 最終的な推奨方針

**V3から借りるのはCLIの非対話実行・stdin・ログと最終回答の分離という実装パターンに限定し、V2の研究制御は維持する。**

実装の中心は「SDKの代わりにsubprocessを一つ呼ぶこと」ではなく、text・画像・schema・履歴・並列処理を既存V2の契約へ接続する共通推論層である。Phase 0でCLIの必要能力を先に検証し、機能削減で問題を回避せず、対応できない条件は明示する。

初版作成時点では仕様策定のみを行った。その後、利用者の実装指示に基づき、以下の変更を実施した。利用者の既存変更（`.gitignore`のSerena関連設定等）は保持している。

## 17. 実装・検証記録（2026-09-27）

### 17.1 実装済みの接続

| 対象 | 実装内容 |
|---|---|
| `inference/contracts.py`, `config.py`, `runtime.py` | JSON契約、role設定、API/CLI分離、明示引数の優先順位、親runtimeとspawn workerのbroker共有 |
| `inference/gateway.py`, `history.py`, `images.py`, `schema.py` | statelessな完全history、実画像bytes/MIME/出現順序、元schema検証、有限retry/repair、正当なnegative判定の保持 |
| `inference/cli.py`, `process.py` | Codex/Claude非対話transport、認証元検査、private cwd、ツール抑制、bounded pipes、deadline/cancel、精密PID同一性による所有process回収 |
| `inference/broker.py` | run単位の独立process、JSON IPC、provider別の共有並列上限、公平queue、quota/auth停止、private artifacts、usage集計、設定・source revision・観測CLI版の追跡 |
| `llm.py`, `vlm.py`, tree backend | 既存戻り値とAPIモードを保持したCLI早期分岐。CLI中のraw SDK迂回を拒否。独立batch/履歴/画像/schemaを接続 |
| ideation・plotting・両writeup・launcher | 共通引数・role・preflight、設定伝播、推論障害の再送出、最終状態保存、所有processだけのcleanup |
| text/VLM review | 直接SDKバイパスを廃止。text reviewの16項目と9整数スコアを厳密検証。reflectionの既存早期終了は維持 |
| tree実行・保存・HTML | 部分Node/ExecutionResult保存、完了済み結果の投入順回収、workspace保持、図の実保存先に整合したURL生成 |
| `inference/doctor.py` | 推論を発行しない診断と、明示的な利用枠消費probeを分離。部分probe成功と全項目成功を区別 |
| `configs/inference.*.yaml`, README, `docs/cli-inference.md` | Codex-only/Claude-only/役割別併用例、導入・制約・停止時の確認方法 |

`bfts_config.yaml`の既定`inference: {}`は旧API動作を保持する。CLI専用設定を使う場合はprofileまたは`--llm-backend`を明示する。launcherがYAML内のCLI設定をAPI既定で無警告に上書きする問題も回帰テストで防止した。APIモードにもprofileのmodel優先順位を反映し、未指定時だけ旧modelを使う。

生成Python用環境は本体gatewayと別の境界として扱う。CLIモードのInterpreterおよびaggregatorから既知LLM credentialsと内部broker情報を除く。APIモードでは既存実験環境を保持する。これはファイルアクセスやネットワークを制限するOS sandboxではない。

### 17.2 設計から具体化した点

- provider実装は別々のファイルではなく`cli.py`に集約した。責務分離とCLIごとのparser/argvは維持している。
- Codex native strict JSON SchemaではV2のoptional/dynamic-key schemaをそのまま表現できないため、native側は`answer_json`文字列のstrict envelope、内側は元schemaのhost検証とした。元schemaを勝手に必須化・null補完・キー固定しない。Claude側は元schemaをnative指定しhostでも検証する。
- 両CLIのtemperature/max_tokensはAPI等価として扱わない。既定は警告＋requested/effective/unsupportedを記録し、strict policyなら送信前に拒否する。
- native JSON不成立とhost schema不適合には同じ有限repair予算を適用する。timeout・broker切断など実行状態が不明な要求は再送しない。
- Codexの診断は`login status`をそのまま読み、host側でAPI認証を拒否する。診断中に`forced_login_method`を変えて既存ログインを失効させない。実推論にはChatGPT認証制約を指定する。
- 現在のprotocol gateはCodex 0.157.1以上（同一major）、Claude Code 2.1.259以上（同一major）。「下限以上なら実機で全能力検証済み」という意味ではない。
- Codex 0.157.1実測で、code-mode hostを意図的に無効にすると`item.type=error`でその診断が届いた。最終回答は成功していたため、当該の正確な既知診断だけを許可し`isolation_notices`へ記録した。その他のerror item、未知の終了状態、一般tool使用は引き続き拒否する。識別子等を除いた実測fixtureをテストへ追加した。
- 生成画像はdecoderの安全なpixel上限もtyped errorとして扱う。wireのbase64込み`max_payload_bytes`と、復元後の`max_materialized_bytes`は別制限として明示した。
- manifestには設定SHA256、全15roleの解決結果、Git revision/dirty/不明理由、観測したCLI版を保存する。Gitは親processで有限時間のread-only確認のみ実施する。要求の入力SHA256は残すが、認証情報・環境変数一覧・prompt/history全文の追加保存は行わない。transportを観測した事実からlive能力やV2全工程を検証済みと推定しない。
- treeの`multiprocessing.Queue`は途中フレーム、`ProcessPoolExecutor`はresult pipe/atexit、`multiprocessing.Event`は待機workerのSIGKILLで停止不能になり得ることを実プロセスで確認した。Interpreterは親側nonblocking socketの増分NDJSON、固定関数の永続worker poolはprivateなatomic JSON task/resultとfile flagに置き換えた。公開Processハンドルのみで有限停止し、pool固有workspace、投入順journal反映、完了済み結果、Node/OmegaConf/Path/idea/nonfinite float/未計測metricの意味を維持する。研究段階や実験数を削る回避は行っていない。

### 17.3 検証の記録

Hypothesis追加は利用者承認済み。テスト用依存は隔離環境へ導入し、グローバル環境は変更していない。`requirements-cli.txt`は推論層/doctor用、`requirements-dev.txt`はCPU-only回帰用、既存`requirements.txt`は全研究用である。

オフライン試験は以下を含む。

- 本番fake executable → 実subprocess → parser → spawn broker/JSON IPC → gateway → LLM/VLM facadeの統合。Codex/Claude両方・混在roleを確認。
- 3独立回答、caller history非破壊、複数画像のbytes/MIME/重複出現順序、optional/dynamic schema、有限repair、negative判定の維持。
- spawn workerと親の同一broker利用、run全体の並列上限、quota後の新規停止、SDK/既知LLM HTTP経路への迂回禁止。
- 実子processのtimeout/cancel、broker SIGKILL後の所有process回収、無関係process生存、PID再利用判定、privateファイル権限。
- 実spawn Interpreterの継続session/環境分離、8MiB途中socketフレーム、fork子のstdout多重書込み、idle workerおよび大きな結果の書込み途中のSIGKILL、有限shutdownと完了結果の保持。Node/OmegaConf/Path/idea/nonfinite floatの往復、seed aggregationの未計測metric、pool間workspace分離。
- APIのmock回帰、査読16項目、両執筆入口・reflection、全入口のrole設定、実際の`--help`の起動。

最終版を固定した全suiteは、重いAPI SDK・PyTorch・PDF依存を入れていない新規venvで **450 passed / 17 skipped**（52.07秒）、既存SDK等を含むproject venvでも **450 passed / 17 skipped**（56.72秒）となった。skipはbackend専用ケース。新規venvの4 warningsは故障注入fixtureでforkを意図的に使うPython警告、project venvの追加1件は既存pyparsingのdeprecationである。新規venvの`pip check`、5入口の実際の`--help`、全Python構文、HTML templateのJavaScript構文、`git diff --check`も成功した。一部のworkflow試験は本番関数をASTから実行して外部サービスだけを置き換えているため、これをV2実機完走とは呼ばない。

利用者の追加承認に基づくCodex実測は**計3 application requestsのみ**実施した。

| 検査 | 結果 |
|---|---|
| 読取専用preflight | Codex 0.157.1、ChatGPT subscription認証を確認。秘密情報は出力していない |
| text | 最終回答と`turn.completed`を取得。当初parserが上記既知診断を拒否したため、修正後に保存済みstream/最終artifactを再解析し整合と診断token形式を確認。再送なし |
| optional/dynamic-key schema | 実モデルで元の期待JSONと一致。成功 |
| 3画像の順序＋schema | 画像の色をランダム化し、ファイル名・promptに正解を含めずに検査。実画像の色と順序が一致。成功 |
| Claude | 初回はCLIがPATH/標準導入先に見つからず未実測。その後の導入・実測は18節を参照。APIへfallbackしていない |

schema/画像の2件は`--probe-check structured_output --probe-check ordered_images`で実行した。そのrunのJSONは`passed_selected_probes`、`live_requests_attempted: 2`であり、全3項目をそのrunで検証したと偽装しない。初回textの旧manifestも失敗記録のまま保持し、過去の結果を書き換えていない。

### 17.4 初回統合時点の未完了の受入条件

1. Claude Codeを導入し利用者がサブスクログインした環境で、tools無効＋schema＋複数画像を実測する。導入・認証設定を無断で変更しない。
2. 対応するCUDA/PyTorch・LaTeX環境で、両CLIの4段階研究・両writeup・最終text/VLM reviewを含む小規模V2全工程を受入試験する。本番研究は今回の3件probeとは別の利用枠/GPU時間を要する。
3. 長文・多画像でのモデル限界、Codex compaction、組織managed policy、複数run運用を実環境で評価する。小規模probeやfake CLIから一般化しない。

任意地点resume、アカウント全体の残枠把握、samplingのAPI完全等価、launcher/broker同時SIGKILL後の全面自動回収は提供しない。停止・保存と再開保証を分離し、これらの制限をREADMEから辿れるようにした。

## 18. Claude実機検証の追補（2026-09-27）

### 18.1 導入・認証

利用者から、Claude CLIの導入と小規模3項目の実推論検証の承認を得た。[公式インストーラー](https://code.claude.com/docs/en/setup)の内容を確認し、管理者権限なしでClaude Code **2.1.283**をユーザー領域の`~/.local/bin/claude`へ導入した。Apple code signatureの検証が成功し、署名者がAnthropic PBCであることを確認した。PATHにも既にユーザー領域のbinが含まれており、shell設定を手動変更する必要はなかった。

公式CLIのsanitizedな`auth status`から、既存の`claude.ai`・`firstParty`・`pro`認証を確認した。利用者の追加ログイン、APIキーの発行、認証ファイルやKeychainからのtoken取り出しは行っていない。API SDKへのfallbackも行っていない。

### 18.2 実機で確認したCLI差異

2.1.283の`--help`は、[公式CLI仕様](https://code.claude.com/docs/en/cli-reference)にある`--system-prompt-file`を独立項目として掲載しない。またglobal flagsを`auth status`の後ろへ置くと`--safe-mode`がunknown optionとなる。同じ隔離flagsとprivateなsystem fileを`auth status`より前へ置くと、非推論の状態確認で受理されることを実測した。必要なflagsを外すのではなく、help非掲載と実際の非対応を区別するactiveな検証、正しい引数順序へ修正した。実際に拒否されるflagは引き続き停止条件である。修正後の本体doctorは`ready_read_only`、推論送信0件で成功した。

doctorには、各要求の直前に期待出力のSHA-256を報告へ記録する処理も加えた。期待値はUTF-8のcanonical JSONでhash化し、textはstrip後のJSON文字列、構造化出力は復号後のanswer値を使う。アルゴリズムと正規化方法も併記し、parserが失敗した場合も当初の期待値と保存済み回答を厳密に再照合できるようにした。nonceや期待objectの本文、認証情報、prompt全文を追加公開するものではない。

追補後の全回帰suiteは、新規隔離venvで**463 passed / 23 skipped**（53.67秒）、project venvでも**463 passed / 23 skipped**（58.79秒）。追加skipはClaude専用ケースをCodexで実行しないためのparameterizationである。Python構文検査と`git diff --check`も成功した。

### 18.3 承認済み3件の実測結果

`python -m ai_scientist.inference.doctor --inference-config configs/inference.claude.yaml --probe`を1回実行し、**3 application requestsすべてが初回成功**した。アプリ側retry/repairは0、API fallbackは無効。追加のCodex推論は行っていない。

| 検査 | 結果 |
|---|---|
| text | ランダムnonceへの完全一致を確認 |
| structured_output | native JSON Schema＋host検証。optional fieldの省略とdynamic-key objectを含め、期待JSONへの完全一致を確認 |
| ordered_images | ランダムな3色の実画像を渡し、色名と添付順序への完全一致を確認 |

doctorの結果は`probes_passed`、`live_requests_attempted: 3`、`all_three_checks_verified: true`、`live_inference_verified: true`。CLIは2.1.283、modelは未指定で、3件ともCLI報告モデルは`claude-opus-5-5`だった。他のモデルや契約にも一般化しない。

成果物はprivateな`.inference/e56ab4ad5bce4cc498fed630e5d7f860/`へ保存した。stdoutで返されたdoctorのJSONも、そのまま`doctor_report.json`へ0600で保存した。元manifestは変更しておらず、3件すべて`transport_status: completed`・`validation_status: accepted`、未完了/未検証IDは空、runは`completed`である。usageはCLI報告値を保存し、実際のサブスク課金額はunknown/nullのまま扱う。

### 18.4 残る受入条件

この実測は推論transportの小規模確認であり、`full_v2_parity_verified: false`のままである。残るのはPyTorch・LaTeXなどの研究用依存が揃った環境での4段階研究、両writeup、最終text/VLM reviewを含む全工程試験、および長文・多画像・managed policy・複数runの評価である。既存のCPU経路を使う場合、CUDAは必須ではない。今回承認された3件を超える実推論や研究実験は、実行先と利用枠を利用者に確認してから行う。

### 18.5 Mac M3での利用方針

利用者のMacはApple M3・メモリ16GB・macOS 15.1で、LaTeXは導入済み、project venvのPyTorchは未導入である。GPU未検出時のCPU経路と生成コードへのCUDA/CPU切替指示が既にあるため、M3でCPU実行するためだけの追加改修は不要と確認した。利用者の意向に合わせ、MPS対応や並列数変更は追加していない。今後は研究用依存の導入と既存CPU経路での小規模な全工程検証が必要であり、CLIの動作確認だけから研究実験の成功・性能を保証しない。
