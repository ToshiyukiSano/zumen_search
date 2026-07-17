"""
register.py - 図面フォルダの一括登録CLI

使い方:
    python register.py ./drawings_folder

指定フォルダ内の PDF/TIFF/PNG/JPG を再帰的に探し、
前処理 → ベクトル化 → ベクトルDB登録 を一括実行する。

付帯情報(材質・板厚など)をCSVで持っている場合は --meta を指定:
    python register.py ./drawings_folder --meta metadata.csv

metadata.csv の形式(1行目ヘッダ必須, file_name でファイル名と突合):
    file_name,part_no,material,thickness,customer
    ABC-001.pdf,ABC-001,SPCC,1.6,○○工業
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from embedder import DrawingEmbedder
from indexer import DrawingIndex
from preprocess import preprocess_for_embedding

SUPPORTED = {".pdf", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def load_metadata(csv_path: str | None) -> dict[str, dict]:
    """CSVからファイル名→付帯情報の辞書を作る。"""
    if not csv_path:
        return {}
    meta = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            name = row.pop("file_name", "").strip()
            if name:
                meta[name] = {k: v.strip() for k, v in row.items() if v.strip()}
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="図面の一括登録")
    parser.add_argument("folder", help="図面フォルダのパス")
    parser.add_argument("--meta", help="付帯情報CSV (任意)", default=None)
    args = parser.parse_args()

    folder = Path(args.folder)
    files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUPPORTED)
    if not files:
        print(f"図面ファイルが見つかりません: {folder}")
        sys.exit(1)

    metadata = load_metadata(args.meta)
    print(f"対象: {len(files)}件  (モデルをロード中...)")

    embedder = DrawingEmbedder()
    index = DrawingIndex(dim=embedder.dim)

    ok, ng = 0, 0
    for i, path in enumerate(files, 1):
        try:
            bw = preprocess_for_embedding(path)
            vec = embedder.embed(bw)
            index.add(path, vec, metadata.get(path.name))
            ok += 1
            print(f"[{i}/{len(files)}] 登録OK: {path.name}")
        except Exception as e:  # 1枚の失敗で全体を止めない
            ng += 1
            print(f"[{i}/{len(files)}] 登録NG: {path.name} ({e})", file=sys.stderr)

    print(f"\n完了: 成功 {ok}件 / 失敗 {ng}件 / DB登録数 {index.count()}件")


if __name__ == "__main__":
    main()
