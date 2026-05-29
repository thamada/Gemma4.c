# 変更履歴

> **注意**:
>   本ドキュメントは変更履歴です。日付はdateコマンドで確認して2026-01-23 12:34:55のように年-月-日 時:分:秒のようにします。
>   最も最新のものから順に並べて記入します。

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
