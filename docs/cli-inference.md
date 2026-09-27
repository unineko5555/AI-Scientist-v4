# サブスクCLIによる推論

## 対象範囲

V2のPython側が研究工程を制御し、本文・コード生成、実験指標抽出、画像評価、段階制御、要約、図の集約、引用、両方式の執筆、査読を共通推論層へ送ります。CLI自体に研究全体や実験コードの実行を委任しません。V3は非対話CLI実装の参考であり、V3の機能削減は取り込んでいません。

`mode: cli` は本体管理のLLM/VLM推論についてAPI SDKへの迂回を禁止します。CLIは公式サービスに通信するので「外部APIやインターネット通信が一切ない」という意味ではありません。Semantic Scholar、データ取得、生成された研究コードの外部通信は別の境界です。CLIモードの生成Pythonには既知のLLM認証情報と内部broker情報を渡しませんが、これは任意コードのネットワークsandboxではありません。研究用環境は引き続き隔離してください。

## 導入と認証

本体の研究実行にはPython・PyTorch・PDF/LaTeXなどの研究用依存が引き続き必要です。既存コードにはCPU実行経路もあるため、CUDAは必須ではありません。Mac M3でもCPU経路を利用できますが、MPSを選択する対応は追加しておらず、研究全工程の実機完走は未検証です。推論層と診断だけなら `python -m pip install -r requirements-cli.txt` で導入できます。POSIX（Linux/macOS）用のCLI runnerで、Windowsネイティブは未対応です。研究実験とCLI推論層の検証範囲は区別してください。

公式CLIを利用者自身が導入・ログインします。アプリは認証ファイルやOAuth tokenを取り出さず、既存の公式ログインとsanitizedな認証状態だけを利用します。

| CLI | 現在のprotocol世代の下限 | 必要な認証 |
|---|---|---|
| Codex | 0.157.1（同一major） | ChatGPTログイン |
| Claude Code | 2.1.259（同一major） | first-party claude.aiサブスクログイン |

これは実推論の動作保証版一覧ではありません。下限以上でも必要flag・認証形式・イベント仕様が合わなければ停止します。モデルごとの画像対応、構造化出力、組織のmanaged policyは実環境で確認が必要です。

2026-09-27の小規模実測では、Claude Code 2.1.283のサブスク認証で3項目のprobeすべてが成功しました。Codex 0.157.1の結果・再解析に関する留保を含め、正確な検証範囲は[development_memo.md](../development_memo.md)の17–18節に記録しています。いずれもV2のGPU/PDF全工程の検証とは別です。

Codexは `codex login`、Claudeは `claude auth login` を利用します。Claudeの `--console`、APIキー認証、Bedrock/VertexはこのCLI profileでは拒否します。認証方式の違いは [Codex公式認証](https://developers.openai.com/codex/auth/)・[Claude公式CLI](https://code.claude.com/docs/en/cli-reference) を参照してください。追加usage購入などの契約設定はアカウント側で管理してください。アプリから追加課金を承認・購入する処理はありませんが、アカウントに既に設定された追加usageをゼロと保証できません。

## 起動

プロジェクトのルートで実行します。

```bash
# 診断のみ。推論リクエストは発行しない。
python -m ai_scientist.inference.doctor --inference-config configs/inference.codex.yaml

# 利用者が明示して実行する小規模推論検証。利用枠を消費する。
python -m ai_scientist.inference.doctor --inference-config configs/inference.codex.yaml --probe

# アイデア生成
python -m ai_scientist.perform_ideation_temp_free \
  --workshop-file ai_scientist/ideas/my_research_topic.md \
  --inference-config configs/inference.codex.yaml

# V2本体
python launch_scientist_bfts.py \
  --load_ideas ai_scientist/ideas/my_research_topic.json \
  --inference-config configs/inference.codex.yaml
```

Claude-onlyなら `configs/inference.claude.yaml`、役割別併用なら `configs/inference.mixed.yaml` に置き換えます。どのroleも使用しないCLIのインストールは不要です。短い指定は `--llm-backend codex_cli` / `--llm-backend claude_cli`。`--llm-model` はCLIの既定model指定です。選択モデルがログイン中の契約で利用できなければ停止します。

`doctor --probe` は使用する各backend/model組でテキスト、optional/dynamic fieldsを含むschema、順序付き複数画像の3要求を送ります。診断時はアプリ側retry/repairを0にし、既定の要求期限を120秒に制限します。CLI内部の推論回数・追加usageまではこの3件と同義ではありません。成功しても長文・managed policy隔離・4研究段階・PDF生成まで保証しません。JSONの `live_inference_verified` と `full_v2_parity_verified` は別項目です。

必要な検査だけ実行する場合は `--probe --probe-check structured_output --probe-check ordered_images` のように指定します。選択した検査のみ成功した場合は `passed_selected_probes` として報告し、全3項目を検証済みにはしません。`--probe-check`だけでは推論を有効にできません。

各probeは送信前に期待出力の`expected_output_fingerprint`を記録します。失敗時も、保存済み回答を追加推論なしで当初の期待値と照合できます。SHA-256・UTF-8・canonical JSONの具体的な方式とtextのstrip規則を報告内へ併記しており、期待値本文や認証情報は表示しません。

## 設定と優先順位

launcherは `bfts_config.yaml` 内の `inference` を既定として読みます。`--inference-config` がある場合はそのprofileに置き換えます。その上で明示された `--llm-backend`・`--llm-model`・既存の `--model_writeup` 等を反映します。省略形のオプション名は拒否します。

roleのmodelは「明示された役割別引数 → profileのrole → profileのdefault → APIモードのみ旧引数/YAMLの既定」の順です。CLIモードでは古い `gpt-*` 等の既定値を継承しません。`null` はCLIの既定を使い、`--model`を付けません。`--llm-backend` はdefaultの上書きであり、profileに明示したrole backendを全て変更するスイッチではありません。API/CLI混在やsubscription-onlyとAPIの組み合わせはエラーです。旧APIに戻す場合はAPI用profileまたは元の空 `inference: {}` を使用してください。

role一覧: `default`, `ideation`, `experiment_code`, `execution_feedback`, `experiment_vision`, `stage_control`, `node_selection`, `summary`, `report`, `aggregation`, `citation`, `writing`, `writing_vision`, `review_text`, `review_vision`。

| 設定 | 既定 | 意味 |
|---|---:|---|
| `providers.<backend>.max_in_flight` | 1 | 同じrunの全process/thread共有のCLI並列上限 |
| `queue_timeout_seconds` | 3600 | 推論を開始する前の待機期限 |
| `request_timeout_seconds` | 1800 | preflightを含む一要求の実行期限 |
| `transient_retry_limit` | 2 | 一時的provider障害の追加試行上限 |
| `schema_repair_limit` | 1 | native/hostの出力契約違反に対する共通補正上限 |
| `max_queue_size` | 64 | 待機要求数の上限 |
| `max_payload_bytes` | 33554432 | 全要求のJSON/base64込みwireサイズ上限 |
| `providers.<backend>.max_materialized_bytes` | 33554432 | 復元したテキスト＋画像bytesの合計上限（wireサイズとは別） |
| `providers.<backend>.max_input_text_bytes` | 524288 | system/historyを含む完全テキスト入力の上限 |
| `providers.<backend>.max_images` | 200 | 画像出現数の上限。モデル側上限は別 |
| `providers.<backend>.max_output_bytes` | 16777216 | CLI出力バッファ上限 |
| `providers.<backend>.max_stdin_bytes` | 67108864 | stdinエンコード後の上限 |

上限超過はエラーにし、画像・履歴を黙って削除しません。個別runの並列制限は、別runや利用者が手動起動したCLIも含むアカウント全体の制限ではありません。

APIのtemperatureとmax_tokensは、両CLIとも同等の制御を適用できません。既定 `unsupported_parameter_policy: warn_and_record` は警告し、requested/effective/unsupportedを記録します。厳密に同じ制御が必要なら `error` を選びます。この場合、旧V2のsampling指定を含む要求は実行前に停止します。出力長・研究品質・samplingがAPIと等価だとは扱いません。

## 履歴・画像・構造化出力

要求ごとに明示historyを完全送信し、CLIのresume/session共有や永続prompt cacheを使いません。batchの各回答は独立した要求です。historyのroleはCLIに与える構造化テキストであり、APIのmessage-role authorityと完全に同一ではありません。

画像は送信前に実bytesを凍結し、実際のMIMEで渡します。同一ファイルを複数回参照した場合も出現IDと順序を保持します。remote URLは自動取得しません。animated画像は明示的に静止frameへ変換する必要があります。

Codexのnative strict schemaとV2のoptional/dynamic-key schemaは表現能力が異なります。Codex側には `{"answer_json": "..."}` のstrict envelopeを指定し、内側を元のV2 JSON Schemaでhost検証します。Claude側は元schemaをnative指定し、返却後も検証します。Codexに元の業務schemaをnative保証させているわけではありません。正当な `valid_plots_received: false` 等を補正で成功へ変えません。査読は16項目・整数スコアの範囲・Accept/Rejectまで検証します。

## 記録と停止

通常、inference artifactsは呼出し元の成果物directory配下の `<run_id>/` に保存されます。明示 `--inference-log-dir PATH` はその既定より優先し、`PATH/<run_id>/` を作ります。run/request directoryは0700、ファイルは0600。**prompt・論文・画像・生成回答を含むprivateデータ**なので公開・commitしないでください。ユーザー定義の保存先は自身でGit除外を設定してください。

- `manifest.json`: run単位の要求、provider並列上限、停止理由、usage、未完了／host検証待ちのID、設定SHA256、解決済みrole、Git revision/dirty状態、観測したCLI版。未確認の能力を検証済みにはしません。
- `requests/<hashed-request-id>/`: 要求メタデータと入力SHA256、最終回答、stdout/stderr、system文、schema、画像、検証結果。transport成功とhost検証成功は区別します。元のprompt/history全量を復元できるresumeログではありません。
- `run_status.json`: launcherの研究工程の終了状態。inference manifestの終了は研究成功と同義ではありません。
- treeのjournal、workspace、`partial_nodes` / `completed_nodes`: 中断時に保持する研究成果。保存は任意地点からの自動resumeを保証しません。

usageはproviderが報告した値のみ。欠測とサブスク課金額はunknown/nullです。cached inputの算入方法をprovider別に残し、inputへ重複加算しません。CLIが返すAPI相当の推定costと実サブスク支出は区別します。

認証切れ・quotaでは新規要求を止め、別provider/APIへ自動fallbackしません。未知の終了状態、timeout、broker切断を成功扱い／無条件再送しません。Ctrl-Cでは所有するCLI/実験processを回収し、成果を保存します。PID/起動IDを検証できないprocessを推測でkillしません。launcherとbrokerの同時SIGKILL、検証できない孤児process group、OSクラッシュまでアプリだけで回収できるとは保証しないため、運用時は専用container/cgroup等の外部監視を併用してください。

実験のstdout/controlは親側nonblocking socket、worker poolのtask/resultはprivate directory内のatomic JSON、停止通知はfile flagで扱います。workerの途中終了で壊れたmultiprocessing QueueやEventを待ち続けないための実装です。結果は従来どおり投入順にjournalへ反映し、完了済みの後続結果も障害時に保存します。workspaceはpoolごとに分離し、GPU予約は所有processの停止を確認してから解放します。

Codexでは検出できたcompactionを拒否しますが、「一切compactionが起きない」という保証はありません。Claudeも設定とイベント検証だけで全managed policyの挙動を証明できません。未知イベントや必要な隔離flagが使えない環境では止まり、権限を緩和して続行しません。

## オフライン回帰テスト

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
```

CLIやAPIアカウントは不要です。fake executableを実processとして起動してbroker・IPC・gateway・facadeを通す試験、履歴/JSON/画像の性質テスト、quota・中断・所有process回収、V2の査読・探索・執筆入口を含みます。一部のtree/入口試験は重い依存を避けて本番関数をASTから実行する試験で、CUDA研究全体の実行試験ではありません。backend専用ケースの他backend分はskipします。
