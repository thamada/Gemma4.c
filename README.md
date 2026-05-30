# Gemma4.c

本リポジトリは、**機械学習フレームワークに依存せず、C 言語だけで Gemma 4 系モデルを推論する**ための実装です。

**PyTorch・TensorFlow・JAX・ONNX Runtime など、ML 向けのユーザランドライブラリ／ランタイムは一切リンクしていません。**  
推論の基準となる実装は **標準 C と `libm`** に加え、CPU 向け最適化として **OpenMP** と **OpenBLAS** を使います。ソースは `gemma4-4b/cpu-blas/main.c` に集約され、ビルドすると **`gemma4-cpu-blas`** ができます。

### なぜライブラリ非依存なのか

一般的な LLM 推論は PyTorch などの高レベルフレームワークを使うと、短いコードで高速に動かせます。一方で、その構成では **計算手順、メモリ配置、量子化レイアウト、RoPE の次元ペアリング** といった低レベルの詳細が、フレームワーク内部に隠れがちです。

本リポジトリでは、あえてその層に依存せず、**GGUF の読み取り、重みの復元、行列演算、Transformer の forward、サンプリングまでを C のコードパスとして明示する** ことを重視しています。既存フレームワークを置き換えるためではなく、推論処理の実体を観察し、検証し、必要に応じて変更できる形で保持するためです。

この方針には、次の意義があります。

- **理解可能性**: モデルファイルから何を読み、どのバッファに置き、どの順序で計算しているかを、ソースコードと `doc/design.md` から直接追跡できる。
- **依存関係の単純化**: Python 環境や大規模な ML スタックを前提にせず、C コンパイラと必要最小限のライブラリで動作経路を確認できる。
- **実験の自由度**: 量子化形式、Attention の実装、サンプリング、チャットテンプレートなど、フレームワークの抽象化に制約されやすい領域を個別に試せる。
- **参照実装としての価値**: 「最小限の構成で Gemma 4 デコーダ推論がどのように成立するか」を示し、llama.cpp 等との比較・検証の基準にできる。

したがって、この実装は最高性能や機能網羅を第一目的とするものではありません。主眼は、LLM 推論の仕組みをブラックボックスにせず、開発者が実装の細部を把握しながら改造できる状態に置くことです。

### 対象モデル

**Gemma 4 E4B Instruct**（`gemma-4-E4B-it-Q4_K_M.gguf`）の **テキストデコーダ** を対象にしています。画像入力や Vision エンコーダは扱いません。プロンプト文字列を入力してテキストを生成する用途に絞っています。

---

## まず何ができるのか

| 機能 | 起動例 | 説明 |
|------|--------|------|
| 単発生成 | `-p "Hello" -n 32` | 1 回のプロンプトに対して応答を生成して終了 |
| 対話モード | `-i` | stdin から複数ターンのチャット（`/quit` で終了） |
| Thinking モード | `--think` | Gemma 4 の `<\|think\|>` / `<\|channel\|>` 形式で推論トレースを生成 |

推論エンジンは **`gemma4-4b/cpu-blas/`** の 1 本です。OpenMP で CPU コアを並列利用し、F32 行列積と Attention を OpenBLAS に任せ、量子化重み（Q4_K / Q5_K / Q6_K）は **Q8_K 活性化 + 整数内積** で処理します。

4B 級モデルでも CPU 実行は重いです。最初の動作確認は `-n 4` など生成トークン数を少なくすると楽です。

---

## ディレクトリ構成

```text
.
├── README.md
├── LICENSE
├── doc/
│   ├── design.md        # 設計仕様（フルスクラッチ実装者向け・必読）
│   └── ChangeLog.md     # 変更履歴・バグ調査経緯
├── tools/               # llama.cpp 比較用ユーティリティ（C++、任意）
└── gemma4-4b/
    ├── Makefile         # `make model` で GGUF 取得・検証
    ├── gguf.txt         # モデル配布 URL
    ├── gemma-4-E4B-it-Q4_K_M.gguf.sha256sum
    └── cpu-blas/
        ├── Makefile
        └── main.c       # 推論エンジン本体
```

推論コードを読むときの入口は **`gemma4-4b/cpu-blas/main.c`** です。設計の全体像と Gemma 4 固有の仕様（ISWA、共有 KV、PLE、Neox RoPE など）は **`doc/design.md`** にまとまっています。

---

## 初心者向け: LLM 推論で何が起きるか

LLM 推論は、大まかには次の流れです。

1. **GGUF ファイルを読む**  
   モデルの重み、語彙、設定値が入った大きなファイルを `mmap` で読み込みます。

2. **プロンプトをトークンに分解する**  
   `"こんにちは"` のような文字列を、モデルが扱える整数 ID の列に変換します。Gemma 4 では BPE と特殊トークン（`<bos>`、`<|turn>user\n` など）を使います。

3. **Prefill（プロンプト処理）**  
   プロンプトの全トークンを Transformer に一度に通し、KV キャッシュを構築します。

4. **Decode（トークン生成）**  
   1 トークンずつ「次に来そうなトークン」を予測し、KV キャッシュを更新しながら繰り返します。

5. **サンプリングする**  
   予測結果（logits）から次のトークンを選びます。`-t`（温度）や `-k`（Top-p）で選び方を調整できます。

6. **トークンを文字列に戻して表示する**  
   選ばれたトークンをテキストとして端末に出します。

このリポジトリの特徴は、この流れを **PyTorch 等に載せず**、**1 つの C ソースの中で追える** ことです。

### Gemma 4 だけのポイント（ざっくり）

初めて LLM をフルスクラッチ実装する場合、Gemma 4 では次の点が特に重要です（詳細は `doc/design.md`）。

| 概念 | ひとことで |
|------|------------|
| **ISWA** | 層ごとに「狭い窓の Attention」と「全履歴の Attention」を切り替える |
| **共有 KV** | 後半 18 層は K/V を再計算せず、前半層のキャッシュを参照する |
| **PLE** | 各層に Per-Layer Embedding を注入する |
| **タイド LM ヘッド** | 独立した `output.weight` がなく、埋め込み行列を転置して logits を出す |
| **Neox RoPE** | 回転する次元ペアが「隣接 `(i, i+1)`」ではなく `(j, j + head_dim/2)` |

これらを 1 つでも間違えると、出力が文字化けしたり Thinking 品質が大きく落ちたりします。`doc/design.md` の **「フルスクラッチ実装アンチパターン一覧」** には、実際にバグとなった誤実装が 12 項目まとめてあります。

---

## 必要なもの

### 共通

- Linux
- `make`
- C コンパイラ（例: `gcc`, `clang`, `cc`）
- `libm`（通常は標準で入っています）
- **OpenBLAS**（`libopenblas-dev` 等）
- **OpenMP ランタイム**（`libgomp1` 等）
- Gemma 4 E4B の GGUF モデルファイル

Ubuntu 系なら、次で基本ツールと依存ライブラリを入れられます。

```bash
sudo apt update
sudo apt install -y build-essential make libopenblas-dev libgomp1
```

または、リポジトリ同梱の Makefile ターゲットを使います。

```bash
cd gemma4-4b/cpu-blas
make openblas    # libopenblas-dev + libgomp1 を apt 導入（要 sudo の場合あり）
```

`cblas.h` が標準パスに無い場合（Debian/Ubuntu の pthread ビルド等）は、ビルド時に `CPPFLAGS` で指定します。

```bash
cd gemma4-4b/cpu-blas
make build CPPFLAGS=-I/usr/include/x86_64-linux-gnu/openblas-pthread
```

実行時は **`OMP_NUM_THREADS`** で CPU 並列度を調整します。OpenBLAS 側は **`openblas_set_num_threads(1)`** で 1 スレッド固定（OpenMP との二重並列化を避ける）です。**`-ffast-math` は Q8_K 量子化で数値が崩れるため Makefile では無効** にしています。

---

## モデルファイルを置く

既定のモデル名は次です。

```text
gemma-4-E4B-it-Q4_K_M.gguf
```

モデルファイルは著作権とファイルサイズの都合により、リポジトリには含めません。`gemma4-4b/gguf.txt` の URL から取得し、`gemma4-4b/` の直下に置きます。推奨は **`make model`**（`wget` + 同梱 `.sha256sum` で検証。既にファイルがありチェックサムが通ればダウンロードをスキップ）です。

```bash
cd gemma4-4b
make model
```

成功時はターミナルにチェックサム検証成功のメッセージが表示されます。

手動で取得する場合:

```bash
cd gemma4-4b
url=$(sed 's|/blob/main/|/resolve/main/|' gguf.txt)
wget -O gemma-4-E4B-it-Q4_K_M.gguf "$url"
sha256sum -c gemma-4-E4B-it-Q4_K_M.gguf.sha256sum
```

配置後、次のようになっていれば準備完了です。

```text
gemma4-4b/
├── Makefile
├── cpu-blas/
│   ├── Makefile
│   └── main.c
└── gemma-4-E4B-it-Q4_K_M.gguf
```

---

## いちばん簡単な実行手順

まず「ビルドできるか」「推論が動くか」を確認します。`-n 4` のように生成トークン数を少なくすると、初回確認が楽です。

```bash
cd gemma4-4b
make model                    # 未取得なら GGUF を取得・検証
cd cpu-blas && make build
./gemma4-cpu-blas ../gemma-4-E4B-it-Q4_K_M.gguf -p "Hello" -n 4
```

うまくいくと、モデル読み込み後にテキストが表示されます。プロンプト区間では stderr に **Prefill progress bar** とスループット要約が出ます。

`Makefile` の `run` を使う場合:

```bash
cd gemma4-4b/cpu-blas
make run PROMPT="Hello, how are you?"
```

---

## ビルドと実行（詳細）

### ビルド

```bash
cd gemma4-4b/cpu-blas
make build
```

成功すると **`gemma4-cpu-blas`** が `cpu-blas/` 直下にできます。

### 実行例

```bash
cd gemma4-4b/cpu-blas

# 日本語プロンプト（先頭空白は add_space_prefix により自動付与）
OMP_NUM_THREADS=8 ./gemma4-cpu-blas ../gemma-4-E4B-it-Q4_K_M.gguf \
  -p "あなたは何者?" -n 64

# 生成を安定させたい（温度を下げ、seed を固定）
OMP_NUM_THREADS=8 ./gemma4-cpu-blas ../gemma-4-E4B-it-Q4_K_M.gguf \
  -p "1文で説明してください: GGUFとは?" \
  -n 32 -t 0.2 -s 42
```

---

## よく使うオプション

| オプション | 例 | 意味 |
|---|---|---|
| `-p` | `-p "Hello"` | 入力プロンプト |
| `-n` | `-n 64` | 最大生成トークン数（1 ターンあたり） |
| `-t` | `-t 0.6` | 温度。低いほど堅め、高いほどランダム（`0` で greedy） |
| `-k` | `-k 0.9` | Top-p。候補を上位確率に絞る |
| `-r` | `-r 1.1` | Repetition penalty（`1.0` で無効） |
| `-s` | `-s 1234` | 乱数シード |
| `-l` | `-l 8192` | 最大シーケンス長（KV キャッシュ上限） |
| `-i` | `-i` | 対話モード |
| `--think` | `--think` | Thinking モード（回答のみ表示） |
| `--show-thinking` | `--show-thinking` | 推論トレースも stderr に表示 |
| `--thinking-budget` | `--thinking-budget 256` | thinking トークン上限 |

まずは短めに試すのがおすすめです。

```bash
./gemma4-cpu-blas ../gemma-4-E4B-it-Q4_K_M.gguf -p "Hello" -n 4
```

---

## 対話モード（`-i`）

`-i` / `--interactive` を付けると、stdin から複数ターンのチャットができます。

```bash
cd gemma4-4b/cpu-blas
OMP_NUM_THREADS=8 ./gemma4-cpu-blas ../gemma-4-E4B-it-Q4_K_M.gguf -i
```

- 各行がユーザーメッセージとして扱われ、Gemma 4 の `<|turn>user\n` / `<|turn>model\n` 形式で履歴が組み立てられます。
- `/quit` または `/exit` で終了します。
- `-p` を併用すると、最初のユーザーメッセージとして使えます。

```bash
./gemma4-cpu-blas ../gemma-4-E4B-it-Q4_K_M.gguf -i -p "短く自己紹介してください"
```

---

## Thinking モード（`--think`）

Gemma 4 の推論トレース（thinking）を生成するモードです。

```bash
# 回答のみ表示（thinking は内部で生成されるが stdout には出さない）
./gemma4-cpu-blas ../gemma-4-E4B-it-Q4_K_M.gguf \
  -p "再帰を説明してください" --think -n 128

# 推論トレースも stderr に表示
./gemma4-cpu-blas ../gemma-4-E4B-it-Q4_K_M.gguf -i --show-thinking
```

`--thinking-budget <n>` で thinking トークン数の上限を指定できます（既定 256、`-1` で無制限）。Thinking モードは **プロンプト組み立てとトークン列の整合** が重要です。詳細は `doc/design.md` の「トークナイザーとチャット形式」を参照してください。

---

## 実装を読きたい人へ

**はじめて LLM をフルスクラッチから実装する** 場合は、次の順番がおすすめです。

1. **この README** — まずビルドと実行を成功させる。
2. **`doc/design.md` の概要と §推論ループ** — Prefill / Decode の全体像を把握する。
3. **`gemma4-4b/cpu-blas/main.c`** — GGUF 読み込みから 1 トークン生成までを追う。
4. **`doc/design.md` の §Attention と RoPE** — Neox RoPE と pos≥1 の数値検証（**必読**）。
5. **`doc/design.md` の §フルスクラッチ実装アンチパターン一覧** — よくある誤実装をチェックリストとして使う。
6. **`doc/ChangeLog.md`** — 実際に起きたバグと修正経緯（特に RoPE 関連）。

`main.c` は 2800 行超と長いですが、おおまかな区切りは次のとおりです。

| 領域 | 探すキーワード |
|------|----------------|
| GGUF 読み込み | `mmap`, `gguf`, tensor |
| トークナイザー | `bpe`, `append_gemma`, `chat_encode` |
| 行列演算 | `mm_`, `cblas_sgemv`, `vec_dot` |
| Attention | `attn_`, `forward_layer` |
| 推論ループ | `prefill`, `decode`, `sample_token` |
| CLI | `main(` |

数値の正しさを確認したい場合は、`tools/` の llama.cpp 比較ユーティリティと `doc/design.md` の「数値検証・回帰テスト」を参照してください（llama.cpp のビルド環境が必要です）。

---

## このリポジトリで扱わないもの

- 学習、ファインチューニング
- GPU / NPU 向け実装（CPU のみ）
- 画像入力、Vision エンコーダ
- サーバ化、Web API 化
- すべての GGUF 量子化形式への汎用対応
- llama.cpp / 公式実装との完全な数値一致保証（ただし `doc/design.md` の手順で突合可能）

目的は、Gemma 4 系 GGUF のテキスト推論を **C** で理解し、実験し、必要に応じて改造できるようにすることです。

---

## よくあるトラブル

### `No such file or directory` と出る

モデルファイルの場所が間違っている可能性があります。

```bash
ls -lh gemma4-4b/gemma-4-E4B-it-Q4_K_M.gguf
```

見つからない場合は `make model` を実行するか、実行時に絶対パスを指定してください。

### ビルドが失敗する／`cblas.h` が見つからない

OpenBLAS 開発パッケージを入れ、必要なら `CPPFLAGS` でヘッダパスを指定してください（上記「必要なもの」参照）。

```bash
cd gemma4-4b/cpu-blas
make openblas
make build
```

### 推論が遅い

正常です。4B 級でも CPU だけでは時間がかかります。`-n 4` などで短く試してください。`OMP_NUM_THREADS` を CPU コア数に合わせると多少改善することがあります。

### 出力が意味不明（同じ文字の連打など）

**`-ffast-math`** を付けてビルドすると Q8_K 量子化内積で数値が崩れます。同梱 `Makefile` では無効化済みです。手元で `CFLAGS` を上書きしている場合は外してください。

### `sha256sum -c` が失敗する

ファイル名または中身が、このリポジトリの想定と違います。ダウンロードが途中で壊れていないか、別量子化のモデルを置いていないか確認してください。

### Thinking モードの品質がおかしい

プロンプトの `<|think|>` / `<|channel|>` 組み立て、RoPE の Neox ペアリング、共有 KV の参照層など、Gemma 4 固有の仕様が絡みます。`doc/design.md` の「Attention と RoPE」「トークナイザーとチャット形式」「アンチパターン一覧」を確認してください。

---

## 片付け

```bash
cd gemma4-4b/cpu-blas && make clean
```

`make clean` では GGUF モデルファイルは削除されません。

---

## 詳細ドキュメント

| ファイル | 内容 |
|----------|------|
| [`doc/design.md`](doc/design.md) | 設計仕様（アーキテクチャ、RoPE、KV、PLE、量子化、チャット形式） |
| [`doc/ChangeLog.md`](doc/ChangeLog.md) | 変更履歴・バグ調査経緯 |

困ったときは、まず `gemma4-4b/` で **`make model`** 済みか、ビルドしたバイナリと実行時に渡しているモデルパスが一致しているかを確認してください。

---

## ライセンス

本リポジトリのソースおよびドキュメントは [Apache License 2.0](LICENSE) に従います。GGUF モデル本体の利用条件は配布元（[unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF) / Google）のライセンスに従います。
