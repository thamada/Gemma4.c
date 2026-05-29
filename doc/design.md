# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴や実装の詳細な変更点については、`ChangeLog.md` を参照してください。本ドキュメントでは、現在のシステムの設計と仕様を記述します。

## 概要

Gemma4.c は **Gemma 4 E4B**（`gemma-4-E4B-it-Q4_K_M.gguf`）を **C 言語のみ**（標準 C11 + `libm`）で推論するためのリポジトリです。PyTorch 等の ML ランタイムには依存せず、GGUF の mmap 読み取り・トークナイズ・Transformer フォワード・サンプリングを単一ソース（`gemma4-4b/cpu/main.c`）に集約しています。

大容量の GGUF モデルファイルは Git 管理外とし、リポジトリには取得手順・整合性検証用チェックサム・CPU 推論実装を含めます。

## ディレクトリ構成

```
Gemma4.c/
├── LICENSE              # Apache License 2.0
├── .gitignore           # Git 管理外ファイルの定義
├── doc/
│   ├── design.md        # 本ドキュメント（設計仕様書）
│   └── ChangeLog.md     # 変更履歴
└── gemma4-4b/
    ├── Makefile         # `make model` によるダウンロード・検証
    ├── gguf.txt         # モデル配布 URL（Hugging Face）
    ├── gemma-4-E4B-it-Q4_K_M.gguf.sha256sum  # SHA256 チェックサム（Git 管理）
    ├── gemma-4-E4B-it-Q4_K_M.gguf            # モデル本体（Git 管理外）
    └── cpu/
        ├── main.c       # CPU 単スレッド推論エンジン
        └── Makefile     # `make build` / `make run`
```

## Git 管理方針

| ファイル | Git 管理 | 備考 |
|----------|----------|------|
| `gemma4-4b/gemma-4-E4B-it-Q4_K_M.gguf` | 外 | `.gitignore` で除外。ローカルに `make model` で取得 |
| `gemma4-4b/gemma-4-E4B-it-Q4_K_M.gguf.sha256sum` | 内 | ダウンロード後の整合性検証に使用 |
| `gemma4-4b/gguf.txt` | 内 | ダウンロード元 URL |
| `gemma4-4b/Makefile` | 内 | モデル取得ターゲット |
| `gemma4-4b/cpu/main.c` | 内 | CPU 推論実装 |
| `gemma4-4b/cpu/Makefile` | 内 | ビルド・実行 |
| `gemma4-4b/cpu/gemma4-cpu` | 外 | ビルド生成物（必要に応じ `.gitignore`） |

`.gitignore` の該当エントリ:

```
gemma4-4b/gemma-4-E4B-it-Q4_K_M.gguf
```

## モデル取得（gemma4-4b）

### 対象モデル

- **ファイル名**: `gemma-4-E4B-it-Q4_K_M.gguf`
- **量子化**: Q4_K_M（線形層は Q4_K / Q5_K / Q6_K、PLE 投影は BF16、norm 等は F32 が混在）
- **配布元**: [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)（URL は `gguf.txt` に記載）

### 取得手順

```bash
cd gemma4-4b
make model
```

### `make model` の動作

1. `gemma-4-E4B-it-Q4_K_M.gguf.sha256sum` の存在を確認する。
2. モデルファイルが既に存在し、チェックサムが一致する場合はダウンロードをスキップする。
3. 存在しない、またはチェックサム不一致の場合、`gguf.txt` の URL（`/blob/main/` を `/resolve/main/` に変換）から `wget` でダウンロードする。
4. ダウンロード後、`sha256sum --check` で整合性を検証する。
5. 検証失敗時は破損ファイルを削除し、再実行を促す。

### Makefile 変数

| 変数 | デフォルト | 説明 |
|------|------------|------|
| `MODEL` | `gemma-4-E4B-it-Q4_K_M.gguf` | 取得・検証対象のモデルファイル名 |

## CPU 推論（gemma4-4b/cpu）

### 実装概要

| 項目 | 内容 |
|------|------|
| ソース | `gemma4-4b/cpu/main.c` |
| ビルド | `make build` → 実行ファイル `gemma4-cpu` |
| 実行 | `make run` または `./gemma4-cpu <model.gguf> [options]` |
| 依存 | C11 コンパイラ、`libm` のみ |
| スレッド | 単スレッド（参照実装・正しさ優先） |

GGUF を `mmap` で読み込み、量子化重み（Q4_K / Q5_K / Q6_K）は **`QK_K=256` ブロック単位**でスタック上に逆量子化しながら GEMV する。全重みの float 一括展開は行わない。

### コマンドラインオプション

| オプション | 既定値 | 説明 |
|------------|--------|------|
| `-p <prompt>` | `Hello, how are you?` | ユーザープロンプト |
| `-n <tokens>` | `256` | 最大生成トークン数 |
| `-t <temp>` | `0.6` | サンプリング温度（`0` で greedy） |
| `-k <topp>` | `0.9` | Top-p サンプリング |
| `-s <seed>` | 時刻 | 乱数シード |
| `-l <len>` | `8192` | 最大シーケンス長（KV キャッシュ上限） |

Prefill 中は stderr に **progress bar**（`Prefill [====...]`）と、完了後に prefill / decode / total の **スループット要約**を出力する。

### ビルド・実行例

```bash
cd gemma4-4b
make model          # 初回のみ GGUF 取得

cd cpu
make build
make run            # 既定 MODEL=../gemma-4-E4B-it-Q4_K_M.gguf

# 例: プロンプトと生成長を指定
./gemma4-cpu ../gemma-4-E4B-it-Q4_K_M.gguf -p "Hello" -n 64
```

### Makefile 変数（cpu）

| 変数 | デフォルト | 説明 |
|------|------------|------|
| `MODEL` | `../gemma-4-E4B-it-Q4_K_M.gguf` | 推論対象 GGUF（`cpu/` からの相対パス） |
| `PROMPT` | `Hello, how are you?` | `make run` 時のプロンプト |
| `CC` | `cc` | C コンパイラ |
| `CFLAGS` | `-O3 -std=c11 -Wall -Wextra ...` | コンパイルフラグ |

## アーキテクチャ（Gemma 4 E4B）

メタデータキーは **`gemma4.*`**。対象 GGUF（42 層）の主要パラメータは次のとおり。

| 項目 | 値 |
|------|-----|
| 埋め込み次元 | 2560 |
| FFN 中間次元 | 10240 |
| レイヤー数 | 42 |
| アテンションヘッド | 8（GQA、KV ヘッド 2） |
| SWA ヘッド次元 | 256（RoPE 256、base 10000） |
| Full ヘッド次元 | 512（RoPE 512、base 1e6、**p-RoPE** で `rope_freqs` 使用） |
| スライディングウィンドウ | 512 |
| 共有 KV 層 | 末尾 18 層（層 0–23 が KV 計算、24–41 は再利用） |
| PLE 次元 | 256（Per-Layer Embeddings） |
| LM ヘッド | **`output.weight` なし**（`token_embd` タイド） |
| Logit softcapping | 30（`tanh` によるクリップ） |

### ISWA（Interleaved Sliding-Window Attention）

`gemma4.attention.sliding_window_pattern` により、層ごとに SWA / Full を切り替える。Full 層は 5, 11, 17, 23, 29, 35, 41 番（0 始まり）。

### 共有 KV キャッシュ

- 層 **0–23**: 自層で K/V を計算し KV キャッシュに書き込む。
- 層 **24–41**: K/V 投影を行わず、次のソース層のキャッシュを参照する。
  - SWA 層 → 層 **22** の KV
  - Full 層 → 層 **23** の KV

### Per-Layer Embeddings（PLE）

各デコーダ層へ補助残差を注入する。

1. `per_layer_token_embd` をトークン ID でルックアップし `√(ple_dim)` でスケール。
2. 主埋め込み（`√dim` スケール済み）を `per_layer_model_proj`（BF16）で射影し `1/√dim` でスケール、`per_layer_proj_norm` で RMSNorm。
3. 上記を加算し `1/√2` でスケール → 層ごとの PLE ベクトル。
4. 各層で `gelu(inp_gate(x)) * ple[l]` → `proj` → `post_norm` → 残差加算。

### フォワードパス（要約）

1. `token_embd` ルックアップ × `√dim`
2. PLE 構築
3. 各層: Pre-Attn RMSNorm → Q/K/V → Q/K norm、V は重みなし RMSNorm → RoPE（Full は `rope_freqs`）→ Attention（SWA は直近 512 トークン）→ 出力投影 → Post-Attn RMSNorm → 残差
4. FFN: Pre-FFN RMSNorm → 並列 GELU FFN（`gelu(gate) * up`）→ Post-FFW RMSNorm → 残差
5. PLE 注入 → `layer_output_scale`（学習可能スカラー、存在する場合）
6. `output_norm` → タイド LM ヘッド → logit softcapping

## トークナイザーとチャット形式

- **方式**: Gemma 4 BPE（`tokenizer.ggml.model` = `gemma4`）
- **前処理**: 空白を U+2581（▁）にエスケープ、改行で分割、GPT-2 バイトフォールバックは使わない
- **特殊トークン**: BOS=2、EOS/EOT=106（`<turn|>`）
- **チャット**: 単一 user ターン + 生成プレフィックス

```text
<bos><|turn>user\n{prompt}<turn|>\n<|turn>model\n
```

出力時は特殊トークンを表示せず、▁ を空白に戻して stdout へ書き出す。

## ライセンス

本リポジトリのソースおよびドキュメントは Apache License 2.0（`LICENSE`）に従います。GGUF モデル本体の利用条件は配布元（Hugging Face / Google）のライセンスに従います。
