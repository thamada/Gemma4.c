# 変更履歴

> **注意**:
>   本ドキュメントは変更履歴です。日付はdateコマンドで確認して2026-01-23 12:34:55のように年-月-日 時:分:秒のようにします。
>   最も最新のものから順に並べて記入します。

## 2026-05-30 05:34:42 — 設計仕様書の更新（/update-doc）

- `doc/design.md`: 本チャットで実装した機能・修正内容を設計仕様として整理
  - 概要にマルチターン対話・Thinking モードを追記
  - CLI オプション表に `-r <penalty>`、`--interactive` 表記、`MAX_GEN_TOKS` / `MAX_LINE` を追記
  - 「サンプリング」節を新設（repetition penalty → top-k 40 → temperature → top-p の順序）
  - 実行例に日本語プロンプト・`-r 1.0` を追加
  - Attention スケール係数 `1.0`（llama.cpp 互換）をフォワードパス説明に明記
  - Thinking モードの既知の制限（llama.cpp との品質差、top-k 固定、共有 KV 設計）を追記

## 2026-05-30 05:31:06 — 対話・Thinking 実装の再調査と llama.cpp 互換性修正

- `gemma4-4b/cpu-blas/main.c` / `gemma4-4b/cpu/main.c`: Gemma 4 チャットテンプレートとトークナイザーの互換性を修正
  - Thinking モードの system ターンを GGUF 内 `tokenizer.chat_template` に合わせて修正
    - 修正前: `<|turn>system\n<|think|><turn|>\n`
    - 修正後: `<|turn>system\n<|think|>\n<turn|>\n`
  - user / model / system ターン開始を、複合特殊トークンが無い場合でも `<|turn>` + role + 改行として組み立てる `append_turn_role()` に集約
  - `<|channel>thought\n` が GGUF 語彙上で単一トークンではなく `<|channel>` + 通常トークン列に分かれるケースへ対応
    - `<|channel>` を Thinking 開始、`<channel|>` を Thinking 終了として検出
    - `<|channel>` 直後の `thought\n` ラベルを表示・履歴保存から除外
    - `--show-thinking` の `--- Thinking ---` / `--- Answer ---` 見出しが重複しないよう状態管理を修正
  - `tokenizer.ggml.add_space_prefix` を GGUF メタデータから読み取るよう追加
    - 既定値は `1`
    - 非空行の先頭へ U+2581（`▁`）を付与してから BPE するよう `gemma_escape_line()` を拡張
    - llama.cpp と同じ `add_space_prefix=true` の挙動に寄せた
  - サンプリング処理を llama.cpp 既定に近づけるため top-k を追加
    - `DEFAULT_TOP_K=40`
    - 従来の全語彙 top-p だけのサンプリングから、top-k 抽出後に temperature / top-p を適用する方式へ変更
  - repetition penalty を実装
    - 既定値は `1.1`
    - 生成済みトークンの logit に対して符号に応じた penalty を適用
    - CLI に `-r <penalty>` を追加し、`-r 1.0` で無効化可能
  - Thinking / answer の出力状態管理を強化
    - `TokenBuf` に生成済みトークンを保持
    - `parse_response()` で Thinking と answer を分離
    - マルチターン履歴には Thinking トレースを保存せず、answer のみ保存

- `doc/design.md`: 実装仕様を更新
  - Thinking テンプレートの `<|think|>` 後に改行を含めるよう修正
  - `<|channel>thought\n` が分割トークンになる場合の境界検出仕様を追記
  - `tokenizer.ggml.add_space_prefix` に従い、非空行先頭へ U+2581 を付与する前処理を追記
  - top-k `40` と repetition penalty `1.1` のサンプリング方針を追記

- 調査・検証
  - llama.cpp で同一 GGUF が正常出力するという前提から、量子化やモデル品質ではなく実装差分を再調査
  - GGUF 内 `tokenizer.chat_template` を確認し、`enable_thinking` 時の system + think ターンの正しい形式を確認
  - GGUF メタデータに `tokenizer.ggml.add_space_prefix=true` が含まれることを確認
  - llama.cpp の Gemma4 実装を確認
    - Gemma4 の `f_attention_scale` は `1.0`
    - `shared_kv_layers` に基づく KV 再利用が使われる
  - `gemma4-4b/cpu-blas`: `make build` 成功
  - `gemma4-4b/cpu`: `make build` 成功
  - `ReadLints`: 関連ファイルに linter エラーなし
  - `cpu-blas` 通常モードで `あなたは何者?` に対して自然な日本語応答を確認
    - 例: `私はGoogle DeepMindによって開発された、オープンウェイトのラージランゲージモデルです。`
  - `cpu-blas` Thinking モードでは `<|channel>` / `<channel|>` の状態遷移は動作するが、llama.cpp と同等品質には未到達

- 調査中に試して戻した変更
  - Attention scale を `1 / sqrt(head_dim)` に変更
    - llama.cpp の Gemma4 実装では `f_attention_scale = 1.0`
    - 実行結果も悪化したため、`attn_scale = 1.0f` に戻した
  - 全 42 層で K/V を計算する変更
    - GGUF には全層の K/V テンソルが存在するが、`shared_kv_layers=18` に基づく再利用と食い違い、通常モードの出力が破綻したため戻した
    - 最終的に `N_LAYER_KV=24`、層 24–41 は SWA 層が 22、Full 層が 23 の KV を参照する実装に戻した

## 2026-05-30 04:06:35 — 対話モード・Thinking モードの追加

- `gemma4-4b/cpu/main.c` / `cpu-blas/main.c`: マルチターン対話と Thinking モードを追加
  - `-i` / `--interactive`: stdin から複数ターンのチャット（`/quit` / `/exit` で終了）
  - `--think`: `<|turn>system\n<|think|>` プレフィックス付きエンコード、`<|channel>thought\n` … `<channel|>` 区間の推論トレース生成
  - `--show-thinking`: 推論トレースを stderr に表示（既定は回答のみ stdout）
  - `ChatHistory` による会話履歴管理（最大 128 ターン）。履歴には回答テキストのみ保存
  - トークナイザー特殊トークン: `turn_system`, `think`, `channel`, `channel_thought`, `channel_end`
- `doc/design.md`: 実行モード・CLI オプション・チャット / Thinking 形式を追記

## 2026-05-30 03:33:13 — ビルド生成物を Git 管理外に設定

- `.gitignore`: `gemma4-4b/cpu/gemma4-cpu` と `gemma4-4b/cpu-blas/gemma4-cpu-blas` を追加
- `doc/design.md`: Git 管理方針の `.gitignore` 記載を更新

## 2026-05-30 03:28:28 — gemma4-4b CPU-BLAS 推論実装の追加

- `gemma4-4b/cpu-blas/` ディレクトリを追加
- `main.c`: `cpu/` と同一デコーダを OpenBLAS + OpenMP で高速化
  - F32: 行帯並列 + `cblas_sgemv`
  - Q4_K / Q5_K: 活性化 Q8_K + 整数内積（AVX2 時 SIMD）
  - Attention: ヘッド単位 BLAS、OpenBLAS は 1 スレッド固定
- `cpu-blas/Makefile`: `make build` / `make run` / `make openblas`（`libopenblas-dev` 等）
- `gemma-4-E4B-it-Q4_K_M.gguf.sha256sum`: Q4_K_M 用チェックサム行を追加（Q8_0 行は維持）
- `doc/design.md`: 二系統 CPU 実装・cpu-blas ビルド手順を追記

## 2026-05-29 18:00:01 — gemma4-4b CPU 推論実装の追加

- `gemma4-4b/cpu/` ディレクトリを追加
- `main.c`: `gemma-4-E4B-it-Q4_K_M.gguf` 向け CPU 単スレッド推論エンジン
  - ISWA（SWA / Full 交互）、共有 KV（層 24–41）、PLE、タイド LM ヘッド、logit softcapping に対応
  - 量子化: Q4_K / Q5_K / Q6_K / BF16 / F32。ブロック単位逆量子化 GEMV
  - Gemma 4 BPE トークナイザー、Gemma 4 チャット形式（`<|turn>user` / `<|turn>model`）
  - Prefill progress bar とスループット要約（stderr）
- `cpu/Makefile`: `make build` / `make run`（既定 `MODEL=../gemma-4-E4B-it-Q4_K_M.gguf`）
- `doc/design.md`: CPU 推論・アーキテクチャ・ビルド手順を追記

## 2026-05-29 17:20:37 — GGUF モデルを Git 管理外に設定

- `.gitignore` を追加し、`gemma4-4b/gemma-4-E4B-it-Q4_K_M.gguf` を Git 管理対象から除外
- モデル本体はローカルで `make model` により取得する運用とする
- チェックサム（`.sha256sum`）および取得手順（`Makefile`, `gguf.txt`）は引き続き Git 管理

## 2026-05-29 — gemma4-4b モデル取得環境の追加

- `gemma4-4b/` ディレクトリを追加
- `Makefile`: `make model` ターゲット（wget ダウンロード + SHA256 検証）
- `gguf.txt`: Hugging Face 上の GGUF 配布 URL
- `gemma-4-E4B-it-Q4_K_M.gguf.sha256sum`: 整合性検証用チェックサムファイル

## 2026-05-30 02:16:39 — Initial commit

- `LICENSE`（Apache License 2.0）を追加
