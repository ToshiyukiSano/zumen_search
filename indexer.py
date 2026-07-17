"""
indexer.py - ベクトルDB (Qdrant ローカルモード) の登録・検索モジュール

QdrantをサーバーレスのローカルモードでPoC利用する。
データは ./qdrant_data/ ディレクトリに永続化されるため、
サーバー構築なしでそのまま使える。本番移行時は
QdrantClient(url="http://社内サーバ:6333") に変えるだけでよい。

付帯情報(品番・材質・板厚など)は payload としてベクトルと一緒に保存し、
フィルタ付き類似検索(例: 材質=SPCC かつ 形状が似ている)を実現する。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

COLLECTION = "drawings"
DATA_DIR = "./qdrant_data"


def _stable_id(file_path: str | Path) -> int:
    """ファイルパスから安定したID(64bit整数)を生成する。
    同じファイルを再登録すると上書きされ、重複登録を防げる。
    """
    h = hashlib.md5(str(Path(file_path).resolve()).encode("utf-8")).hexdigest()
    return int(h[:15], 16)  # Qdrantのunsigned int範囲に収める


class DrawingIndex:
    """図面ベクトルの登録・類似検索を担うクラス。"""

    def __init__(self, dim: int, data_dir: str = DATA_DIR):
        self.client = QdrantClient(path=data_dir)  # ローカルモード(永続化)
        self.dim = dim
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(
        self,
        file_path: str | Path,
        vector: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """図面1件を登録する。metadataには品番・材質・板厚などを渡す。"""
        point_id = _stable_id(file_path)
        payload = {"file_path": str(Path(file_path).resolve())}
        if metadata:
            payload.update(metadata)
        self.client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=point_id, vector=vector.tolist(), payload=payload)],
        )
        return point_id

    def search(
        self,
        vector: np.ndarray,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        exclude_file: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """類似検索。filters={"material": "SPCC"} のように付帯情報で絞り込める。

        戻り値: [{"score": 0.93, "file_path": "...", ...payload}, ...] 類似度降順
        """
        qfilter = None
        if filters:
            qfilter = Filter(
                must=[
                    FieldCondition(key=k, match=MatchValue(value=v))
                    for k, v in filters.items()
                    if v not in (None, "")
                ]
            )
        hits = self.client.query_points(
            collection_name=COLLECTION,
            query=vector.tolist(),
            limit=top_k + 1,  # 自分自身がヒットする分を考慮して+1
            query_filter=qfilter,
        ).points

        exclude = str(Path(exclude_file).resolve()) if exclude_file else None
        results = []
        for h in hits:
            payload = h.payload or {}
            if exclude and payload.get("file_path") == exclude:
                continue  # 検索元と同じファイルは除外
            results.append({"score": round(float(h.score), 4), **payload})
        return results[:top_k]

    def count(self) -> int:
        """登録済み図面数を返す。"""
        return self.client.count(collection_name=COLLECTION).count
