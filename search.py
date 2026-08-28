"""
search.py - 類似図面検索CLI

使い方:
    python search.py 検索したい図面.pdf
    python search.py 検索したい図面.dxf --top 10 --material SPCC
"""

from __future__ import annotations

import argparse

from embedder import DrawingEmbedder
from indexer import DrawingIndex
from preprocess import preprocess_for_embedding


def main() -> None:
    parser = argparse.ArgumentParser(description="類似図面検索")
    parser.add_argument("query", help="検索したい図面ファイル")
    parser.add_argument("--top", type=int, default=10, help="表示件数 (default:10)")
    parser.add_argument("--material", help="材質で絞り込み (例: SPCC)")
    parser.add_argument("--customer", help="得意先で絞り込み")
    args = parser.parse_args()

    embedder = DrawingEmbedder()
    index = DrawingIndex(dim=embedder.dim)

    bw = preprocess_for_embedding(args.query)
    vec = embedder.embed(bw)

    filters = {}
    if args.material:
        filters["material"] = args.material
    if args.customer:
        filters["customer"] = args.customer

    results = index.search(
        vec, top_k=args.top, filters=filters or None, exclude_file=args.query
    )

    if not results:
        print("類似図面が見つかりませんでした(登録数が0の可能性があります)")
        return

    print(f"\n=== 類似図面 上位{len(results)}件 ===")
    for rank, r in enumerate(results, 1):
        info = "  ".join(
            f"{k}={v}" for k, v in r.items() if k not in ("score", "file_path")
        )
        print(f"{rank:2d}. 類似度 {r['score']:.3f}  {r['file_path']}")
        if info:
            print(f"     {info}")


if __name__ == "__main__":
    main()
