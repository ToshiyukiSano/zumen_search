# AI類似図面検索 PoC プロトタイプ

設計資料のPhase 1に対応する、前処理+ベクトル化+検索+簡易UIの一式です。

## ファイル構成

| ファイル | 役割 |
|----------|------|
| `preprocess.py` | 図面の画像化・二値化・傾き補正・寸法線軽減・表題欄切り出し |
| `embedder.py` | DINOv2による図面画像のベクトル化(384次元) |
| `indexer.py` | ベクトルDB(Qdrantローカルモード)への登録・類似検索・絞り込み |
| `register.py` | 図面フォルダの一括登録CLI |
| `search.py` | コマンドラインでの類似検索 |
| `app.py` | Streamlitによる簡易Web UI(現場評価用) |
| `requirements.txt` | 依存パッケージ一覧 |

## セットアップ

```bash
# Python 3.10以上を推奨
pip install -r requirements.txt
```

- 初回実行時にDINOv2モデル(約90MB)が自動ダウンロードされます
- GPUは不要です(あれば自動で使用され高速化します)
- ベクトルDBはサーバー不要で `./qdrant_data/` に自動保存されます

## 使い方

### 1. 図面の一括登録

```bash
python register.py ./図面フォルダ
```

材質・板厚などの付帯情報があればCSVで渡せます:

```bash
python register.py ./図面フォルダ --meta metadata.csv
```

`metadata.csv` の形式:

```csv
file_name,part_no,material,thickness,customer
ABC-001.pdf,ABC-001,SPCC,1.6,○○工業
ABC-002.pdf,ABC-002,SUS304,2.0,△△製作所
```

### 2. コマンドラインで検索

```bash
python search.py 検索したい図面.pdf
python search.py 検索したい図面.pdf --top 10 --material SPCC
```

### 3. Web UIで検索(現場評価用)

```bash
streamlit run app.py
```

ブラウザが開くので、図面をアップロードして「類似図面を検索」を押すだけです。

## 精度チューニングのポイント

1. **寸法線の軽減強度**: `preprocess.py` の `suppress_thin_lines(kernel_size=2)` を2〜3で調整。消しすぎると外形線が欠けます
2. **モデルサイズ**: 精度不足なら `embedder.py` の `MODEL_NAME` を `dinov2_vitb14`(768次元)に変更
3. **評価方法**: 設計資料7章の通り、ベテランが選んだ正解セットでTop-5正解率を毎回計測してから変更の良し悪しを判断してください

## 本番移行時の変更点

- Qdrantをサーバーモードで立てる場合: `indexer.py` の `QdrantClient(path=...)` を `QdrantClient(url="http://社内サーバ:6333")` に変更するだけ
- 表題欄のAI-OCR(品番・材質の自動抽出)はPhase 2で追加予定。`preprocess.py` の `crop_title_block()` が切り出した画像をマルチモーダルAI APIに渡す構成を想定しています
