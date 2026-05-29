# 設計仕様書

> **注意**: 本ドキュメントは設計仕様書です。変更履歴や実装の詳細な変更点については、`ChangeLog.md` を参照してください。本ドキュメントでは、現在のシステムの設計と仕様を記述します。

## 概要

Gemma4.c は Gemma 4 モデルを C 言語で扱うためのリポジトリです。大容量の GGUF モデルファイルは Git 管理外とし、リポジトリには取得手順と整合性検証用のチェックサムのみを含めます。

## ディレクトリ構成

```
Gemma4.c/
├── LICENSE              # Apache License 2.0
├── .gitignore           # Git 管理外ファイルの定義
├── doc/
│   ├── design.md        # 本ドキュメント（設計仕様書）
│   └── ChangeLog.md     # 変更履歴
└── gemma4-4b/           # Gemma 4 4B モデル取得用
    ├── Makefile         # `make model` によるダウンロード・検証
    ├── gguf.txt         # モデル配布 URL（Hugging Face）
    ├── gemma-4-E4B-it-Q4_K_M.gguf.sha256sum  # SHA256 チェックサム（Git 管理）
    └── gemma-4-E4B-it-Q4_K_M.gguf            # モデル本体（Git 管理外）
```

## Git 管理方針

| ファイル | Git 管理 | 備考 |
|----------|----------|------|
| `gemma4-4b/gemma-4-E4B-it-Q4_K_M.gguf` | 外 | `.gitignore` で除外。ローカルに `make model` で取得 |
| `gemma4-4b/gemma-4-E4B-it-Q4_K_M.gguf.sha256sum` | 内 | ダウンロード後の整合性検証に使用 |
| `gemma4-4b/gguf.txt` | 内 | ダウンロード元 URL |
| `gemma4-4b/Makefile` | 内 | モデル取得ターゲット |

`.gitignore` の該当エントリ:

```
gemma4-4b/gemma-4-E4B-it-Q4_K_M.gguf
```

## モデル取得（gemma4-4b）

### 対象モデル

- **ファイル名**: `gemma-4-E4B-it-Q4_K_M.gguf`
- **量子化**: Q4_K_M
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

## ライセンス

本リポジトリのソースおよびドキュメントは Apache License 2.0（`LICENSE`）に従います。GGUF モデル本体の利用条件は配布元（Hugging Face）のライセンスに従います。
