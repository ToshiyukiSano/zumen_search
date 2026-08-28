# AI類似図面検索 PoC プロトタイプ (v2: DXF対応版)

前処理+ベクトル化+検索+簡易UIの一式。スキャン図面(PDF/TIFF/画像)に加え、
CADデータ(DXF)にも対応。両者を混在させて登録・相互検索できます。

## v2での変更点

- **DXF対応を追加** (`dxf_loader.py` 新規)
  - 寸法(DIMENSION)・文字(TEXT/MTEXT)・引出線をエンティティ型で確実に除去し、形状だけをレンダリング
  - レイヤー名(「寸法」「注記」等を含む)でも除外可能。社内ルールに合わせて `EXCLUDE_LAYER_KEYWORDS` を調整
  - 図面内テキストの抽出機能付き(表題欄の品番・材質をOCRなしで取得する足がかり)
  - DXFはデスキュー・細線除去をスキップ(元々傾きゼロ・除去済みのため)
- レビュー指摘反映済み: deskew角度補正の修正 / qdrant-client>=1.10 / 一時ファイル削除

## ファイル構成

| ファイル | 役割 |
|----------|------|
| `preprocess.py` | 画像系の前処理 + DXFへの振り分け |
| `dxf_loader.py` | DXFのレンダリング・寸法除去・テキスト抽出 |
| `embedder.py` | DINOv2によるベクトル化(384次元) |
| `indexer.py` | ベクトルDB(Qdrantローカルモード)登録・検索・絞り込み |
| `register.py` | 一括登録CLI(DXFはテキストも自動保存) |
| `search.py` | コマンドライン検索 |
| `app.py` | Streamlit簡易Web UI |

## セットアップ

```bash
pip install -r requirements.txt
```

## 使い方

```bash
# 登録(PDF/TIFF/画像/DXF混在でOK)
python register.py ./図面フォルダ --meta metadata.csv

# 検索
python search.py 検索したい図面.dxf --material SPCC

# Web UI
streamlit run app.py
```

## DXFに関する注意

- **DWGは直接読めません**。ODA File Converter(無料)でDXFに一括変換してから登録してください
- 図枠・表題欄までレンダリングに含まれてしまう場合は、`dxf_loader.py` の `EXCLUDE_LAYER_KEYWORDS` に自社の図枠レイヤー名を追加してください
- `extract_dxf_text()` の抽出結果をマルチモーダルAI/LLMに渡して「品番・材質・板厚をJSONで」と構造化させるのがPhase 2のAI-OCR相当機能になります(DXFの場合は画像OCRより確実)

## 動作確認済みの内容(開発環境でのテスト)

- テストDXF(外形+穴+DIMENSION寸法+注記文字+レイヤー分け)で、寸法・注記の除去レンダリングを確認(黒画素数 18998→5481 に減少)
- preprocess → サムネイル生成 → ベクトルDB登録 → スキャン画像と混在での類似検索、の一連の流れを確認
- テキスト抽出で「t1.6 SPCC」「品番: BRK-2026-001」を正しく取得
