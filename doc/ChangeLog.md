# 変更履歴

> **注意**:
>   本ドキュメントは変更履歴です。日付はdateコマンドで確認して2026-01-23 12:34:55のように年-月-日 時:分:秒のようにします。
>   最も最新のものから順に並べて記入します。

## 2026-05-30 08:23:06 — Thinking 品質調査: フォワード pass 量子化誤差が原因（調査経緯の詳細）

### 最終結論

`--think` モードで thinking / answer が llama.cpp と比べて劣化する主因は、**テンプレート・トークナイザー・サンプラーではなく、Q4_K×Q8_K 量子化 matmul の累積誤差によるフォワード pass の logits 乖離**である。近接した logits 候補の argmax が decode 中に逆転し、step 18 付近で生成軌道が分岐、step 25 以降で劣化ループ（`3689` 等）に入る。`<channel|>`（token 101）に到達できず thinking が終わらない事象は、この劣化の二次症状である。

---

### 1. きっかけ（症状）

- **通常モード**（`--think` なし、13 prompt tokens）: 日本語で正常応答（例: 「私は Google DeepMind によって開発された…」）
- **`--think` モード**（20 prompt tokens）: 英語の崩れた thinking → answer も無意味な記号列
- **`--show-thinking`**: 生成ロジックには触れず表示のみ。品質問題の本体は `--think` 側

同一 GGUF（`gemma-4-E4B-it-Q4_K_M.gguf`）を llama.cpp で `--reasoning on` すると thinking → answer まで正常 → **モデル・量子化形式そのものではなく実装差**と判断。

---

### 2. 初期仮説と切り分け

#### 2.1 テンプレート / プロンプト形式

| 仮説 | 検証 | 結果 |
|------|------|------|
| `<\|channel>thought\n` がプロンプトに prefilled されていない | llama.cpp Jinja テンプレート・`test-chat.cpp` を確認 | **`enable_thinking=true` 時は `<\|turn>model\n` で終わり、channel は decode 1 トークン目として生成される設計**。Gemma4.c も同様 → **非原因** |
| system + `<\|think|>` ターンの改行不足 | ChangeLog 05:31 時点で修正済み | テンプレート構造は llama と整合 |
| プロンプト token ID 不一致 | `--dump-prompt` vs `llama-tokenize` / `tools/llama_dump_gen` | Gemma4 `--think` の **20 token 列は llama chat template と一致** |

Gemma4 `--dump-prompt` 出力（20 tokens）:

```text
2 105 9731 107 98 107 106 107 105 2364 107 51953 169845 237457 236881 106 107 105 4368 107
```

※ 手書き比較用プロンプトファイル（21 tokens）は user 行末 `\n<turn|>` の改行位置が異なり、**比較用 artifact の問題**だった。Gemma4 自身のエンコードは一貫。

#### 2.2 サンプラー

| 変更 | 内容 | 結果 |
|------|------|------|
| repetition penalty | 既定 `1.1` → **`1.0`（無効）**、`-N 64` ウィンドウ追加 | thinking 長生成への過剰 penalize を除去したが **品質問題は未解決** |
| greedy 高速経路 | `logit_softcapping > 0` 時は `mm_argmax_row` を無効化 | softcap スキップバグを修正 |
| 改行トークン | 改行のみ塊の語彙 lookup（llama PR 21343 相当） | プロンプト token 一致に寄与 |

→ サンプラー修正だけでは thinking 品質は直らない。**greedy（`-t 0`）でも軌道が分岐**するため、サンプラー以前の logits 差が疑われる。

#### 2.3 フォワードパス構造（試して revert 済み）

| 試行 | 結果 |
|------|------|
| Attention scale を `1/√head_dim` に変更 | llama Gemma4 は `f_attention_scale=1.0`。**悪化のため revert** |
| 全 42 層で K/V を独立計算 | `shared_kv_layers=18` 設計と食い違い。**通常モードも破綻のため revert** |
| MoE 層未実装 | E4B GGUF に `ffn_gate_inp` なし。**通常モード正常** → thinking 専用原因ではない |

確定した KV 設計: `N_LAYER_KV=24`、層 24 以降は llama `layer_reuse_cb` と同様（SWA→22、Full→23 を参照）。

---

### 3. 決定的な実験: greedy token 列の step-by-step 比較

プロンプト `あなたは何者?`、`--think`、`-t 0 -s 42`、同一 GGUF。

**生成 token 列**（`tools/llama_dump_gen` vs `--dump-gen`）:

| gen step | llama.cpp | Gemma4.c | 備考 |
|----------|-----------|----------|------|
| 0–17 | 一致 | 一致 | 先頭 `<\|channel>thought` 相当まで一致 |
| **18** | **236764** | **236787** | **初の分岐** |
| 19–24 | 一致 | 一致 | KV 差があっても argmax が偶然一致 |
| **25** | **2267** | **3689** | **恒久分岐・劣化開始** |
| 以降 | 正常 thinking 継続 | `3689` ループ等 | `<channel|>` 未到達 |

---

### 4. Teacher forcing による因果の特定

`--force-gen`（cpu-blas）で llama の token 列を Gemma4 に強制入力。

| 実験 | 期待 | 実際（Gemma4） |
|------|------|----------------|
| llama 0–24 token 強制後の 26 番目 | 2267 | **3689** |
| llama 0–17 token 強制後、18 番目以降 free | 236764 から再合流 | **236787**（llama と同じ分岐点） |

→ **同一 KV 状態（同一 token 履歴）でも次 token の logits が llama と異なる** = KV バグやサンプラーではなく **forward 実装の数値差**。

---

### 5. Logits ダンプ（`--dump-logits-at` / `tools/llama_dump_logits`）

**公平比較**: 同一 20 token プロンプト（`--tokens`）+ 同一 gen prefix（`--force`）。

#### gen step 18（prefix 18 token 強制後）

| 順位 | llama.cpp | Gemma4.c |
|------|-----------|----------|
| 1 | 623=23.62 | **236787=24.29** |
| 2 | 236764=23.37 | 528=23.20 |
| 3 | 236787=22.14 | **236764=23.02** |

#### gen step 25（llama prefix 26 token 強制後）

| 順位 | llama.cpp | Gemma4.c |
|------|-----------|----------|
| 1 | **2267=26.98** | **3689=25.14** |
| 2 | 3689=25.05 | 15938=24.47 |
| 6 | — | **2267=23.08** |

正解 token（2267）の logit だけ **約 4 ポイント** llama より低く、argmax が逆転。

#### gen step 0（prefill 直後）

- 両者とも argmax **100**（`<|channel>`）— ここでは一致
- final hidden L2 norm: llama **251.5** vs Gemma4 **228.8**（**prefill 時点で約 9% drift**）

---

### 6. Hidden state 指紋（`--dump-hidden-at`）

pos 37（gen step 18 相当）の層出力 sum / L2 / max を層ごとに記録。pos 19（prefill 終了）と比較すると、**浅い層（layer 2 付近）から統計量が乖離**し、深層に至るまで増幅。最終層 pre_final_norm（pos 44, step 25）: sum=77.93, l2=55.10。

---

### 7. 除外した原因

| 候補 | 根拠 |
|------|------|
| AVX2 `vec_dot_q4_K_q8_K` バグ | `-DGEMMA4_USE_GENERIC_DOT` で generic 実装に切替 → **logits 不変** |
| `--show-thinking` | `PrintMode` 切替のみ |
| repetition penalty 既定 1.1 | 1.0 化後も greedy 分岐 |
| プロンプト token 列 | 20 token 列は llama と一致 |
| attn_scale / KV 独立化 / MoE | 上記参照 |

---

### 8. thinking が終わらない理由（二次症状）

- 劣化ループ中 **`<channel|>`（101）が出ない**
- `-n 8192` 等で thinking が延々続く
- 対策として **`--thinking-budget 256`（既定）** で `<channel|>` を強制し answer へ移行（安全装置）。answer 品質も依然不良

---

### 9. 追加した調査用コード・ツール

| パス | 用途 |
|------|------|
| `gemma4-4b/cpu-blas/main.c` | `--dump-prompt`, `--dump-gen`, `--dump-logits-at`, `--dump-hidden-at`, `--force-gen`, `--thinking-budget`, `-DGEMMA4_USE_GENERIC_DOT` |
| `tools/llama_dump_gen.cpp` | llama greedy token ID ダンプ |
| `tools/llama_dump_logits.cpp` | llama logits / hidden ダンプ（`--tokens`, `--force`, `-d step`） |
| `tmp/llama.cpp/` | 比較用 llama ビルド・`llama-eval-callback` |

- `doc/design.md`: デバッグオプション・`tools/`・Thinking 品質の根本原因を追記

---

### 10. 次の修正方針（未実装）

1. **ggml の Q4_K×Q8_K 内積カーネルをそのまま利用**（llama.cpp と数値一致を目指す）
2. **LM head または最終数層の F32 dequant matmul**（argmax 逆転の直接原因を除去）
3. **層単位 diff**（llama `l_out-N` eval-callback vs Gemma4 `--dump-hidden-at`）で最初に大きく乖離する層を特定

---

## 2026-05-30 08:42:04 — F32 matmul 切り分け: Q4/Q5 量子化 matmul は主因ではない

### 実施内容

- `gemma4-4b/cpu-blas/main.c`: デバッグ用 **`--f32-matmul`** を追加
  - Q4_K / Q5_K 重みを **F32 dequant 後に matmul**（`mm_quant_rows` 経路）。Q8_K 活性化量子化をバイパス
- `tools/llama_dump_logits.cpp`: teacher forcing 後の **hidden（embeddings API）** ダンプを安定化

### 同一 prefix（20 prompt + llama 25 gen token 強制）gen step 25 の比較

| 指標 | llama.cpp | Gemma4.c（Q8 既定） | Gemma4.c（`--f32-matmul`） |
|------|-----------|---------------------|----------------------------|
| top-1 logit | **2267=26.98** | 3689=25.14 | 3689=25.14 |
| 2267 | 1 位 | 6 位（23.08） | 5 位（23.16） |
| pre_final L2 | — | 55.10 | 55.29 |
| final hidden L2（llama embeddings / Gemma4 final_norm） | **300.44** | 193.17 | （ほぼ同値） |

### 結論（前回仮説の修正）

- **`--f32-matmul` でも logits / hidden はほぼ不変** → Gemma4 内部の Q4_K×Q8_K 内積と F32 dequant 経路は **同等精度** であり、**Q4 量子化 matmul 単体が llama との差の主因ではない**
- llama との差は **Attention / KV cache / RoPE / PLE / Q6_K 経路** など、フォワード pass の別箇所に起因する可能性が高い
- gen step 25 時点で llama hidden L2（300.44）と Gemma4 final_norm L2（193.17）に **大きな乖離** → 層単位 diff の継続が必要

### 次の作業

- llama `llama-eval-callback` と Gemma4 `--dump-hidden-at` で **同一 token 列・同一 pos の層出力** を突合
- Attention（`cblas_sgemv` + KV レイアウト + SWA ウィンドウ + KV 再利用）を llama `llama_kv_cache_iswa` と照合

### 関連 ChangeLog エントリ

- `2026-05-30 06:46:19` — サンプラー修正・プロンプト token 一致確認（フォワード乖離の**推定**段階）
- `2026-05-30 07:12:05` — thinking budget（無限 thinking の**暫定回避**）
- `2026-05-30 05:31:06` — テンプレート・トークナイザー修正、フォワード試行の revert 記録

## 2026-05-30 07:22:29 — cpu 参照実装を削除し cpu-blas に一本化

- `gemma4-4b/cpu/` を削除（`main.c`, `Makefile`）。推論実装は `cpu-blas/` のみとする
- `.gitignore`: `tmp/*` を追加（llama.cpp 比較等の一時ファイル用）
- `doc/design.md`: 二系統 CPU 実装の記述を削除し、`cpu-blas` 単一構成に更新

## 2026-05-30 07:12:05 — Thinking budget 追加（無限 thinking 対策）

- `gemma4-4b/cpu-blas/main.c`
  - **`--thinking-budget <n>`** を追加（`--think` 有効時の既定 **256**、`-1` = 無制限）
  - thinking 区間が上限に達したら `<channel|>` を強制挿入し answer フェーズへ移行
  - stderr に `[thinking budget N tokens reached; forcing <channel|>]` を表示
- `doc/design.md`: CLI と Thinking モード説明を更新

## 2026-05-30 06:46:19 — Thinking 品質調査: サンプラー修正と llama.cpp 比較

- `gemma4-4b/cpu-blas/main.c`: llama.cpp 互換のサンプラー修正
  - repetition penalty 既定値を `1.1` → **`1.0`（無効）** に変更（llama.cpp 既定と一致）
  - **`penalty_last_n=64`** ウィンドウを追加（`-N <n>`、`0` = 全生成トークン）
  - greedy 高速経路（`mm_argmax_row`）が logit softcapping をスキップしないよう修正（`logit_softcapping > 0` 時は無効化）
  - Gemma4 改行トークン: 改行のみの文字列を語彙直接 lookup（llama.cpp PR 21343 相当）
  - デバッグ用 **`--dump-prompt`** / **`--dump-gen`** を追加
- `doc/design.md`: CLI 既定値・既知の制限を更新
- 調査結果
  - `--think` 時のプロンプト token ID 列は llama.cpp `llama-tokenize` と **完全一致**（20 tokens）
  - 品質劣化はテンプレート／トークナイズではなく、**長い decode 中の logits 乖離**が原因と推定
  - greedy でも約 40 トークン以降で llama.cpp と生成軌道が分岐（通常モードは 40 トークン程度で問題なし）
  - repetition penalty 既定無効化は thinking 長生成への過剰 penalize を除去

## 2026-05-30 05:34:42 — 設計仕様書の更新（/update-doc）

- `doc/design.md`: 本チャットで実装した機能・修正内容を設計仕様として整理
  - 概要にマルチターン対話・Thinking モードを追記
  - CLI オプション表に `-r <penalty>`、`--interactive` 表記、`MAX_GEN_TOKS` / `MAX_LINE` を追記
  - 「サンプリング」節を新設（repetition penalty → top-k 40 → temperature → top-p の順序）
  - 実行例に日本語プロンプト・`-r 1.0` を追加
  - Attention スケール係数 `1.0`（llama.cpp 互換）をフォワードパス説明に明記
  - Thinking モードの既知の制限（llama.cpp との品質差、top-k 固定、共有 KV 設計）を追記

## 2026-05-30 05:31:06 — 対話・Thinking 実装の再調査と llama.cpp 互換性修正

- `gemma4-4b/cpu-blas/main.c`: Gemma 4 チャットテンプレートとトークナイザーの互換性を修正
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

- `gemma4-4b/cpu-blas/main.c`: マルチターン対話と Thinking モードを追加
  - `-i` / `--interactive`: stdin から複数ターンのチャット（`/quit` / `/exit` で終了）
  - `--think`: `<|turn>system\n<|think|>` プレフィックス付きエンコード、`<|channel>thought\n` … `<channel|>` 区間の推論トレース生成
  - `--show-thinking`: 推論トレースを stderr に表示（既定は回答のみ stdout）
  - `ChatHistory` による会話履歴管理（最大 128 ターン）。履歴には回答テキストのみ保存
  - トークナイザー特殊トークン: `turn_system`, `think`, `channel`, `channel_thought`, `channel_end`
- `doc/design.md`: 実行モード・CLI オプション・チャット / Thinking 形式を追記

## 2026-05-30 03:33:13 — ビルド生成物を Git 管理外に設定

- `.gitignore`: `gemma4-4b/cpu-blas/gemma4-cpu-blas` を追加
- `doc/design.md`: Git 管理方針の `.gitignore` 記載を更新

## 2026-05-30 03:28:28 — gemma4-4b CPU-BLAS 推論実装の追加

- `gemma4-4b/cpu-blas/` ディレクトリを追加
- `main.c`: Gemma 4 E4B デコーダを OpenBLAS + OpenMP で実装
  - F32: 行帯並列 + `cblas_sgemv`
  - Q4_K / Q5_K: 活性化 Q8_K + 整数内積（AVX2 時 SIMD）
  - Attention: ヘッド単位 BLAS、OpenBLAS は 1 スレッド固定
- `cpu-blas/Makefile`: `make build` / `make run` / `make openblas`（`libopenblas-dev` 等）
- `gemma-4-E4B-it-Q4_K_M.gguf.sha256sum`: Q4_K_M 用チェックサム行を追加（Q8_0 行は維持）
- `doc/design.md`: CPU 推論・cpu-blas ビルド手順を追記

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
