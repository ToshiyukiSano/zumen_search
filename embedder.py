"""
embedder.py - 図面画像のベクトル化モジュール

DINOv2 (Meta公開の自己教師あり学習モデル) を使い、
前処理済み図面画像を数値ベクトル(384次元)に変換する。

- モデルは初回実行時に自動ダウンロードされる(約90MB, dinov2_vits14)
- CPUでも1枚あたり1秒前後で処理可能。GPUがあれば自動で使用する
- 精度を上げたい場合は MODEL_NAME を "dinov2_vitb14"(768次元) に変更
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as T

MODEL_NAME = "dinov2_vits14"  # small版: 384次元, 軽量
INPUT_SIZE = 518              # DINOv2の推奨入力サイズ(14の倍数)


class DrawingEmbedder:
    """図面画像 → ベクトル 変換器。プロセス内で1インスタンスを使い回す。"""

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # torch.hub 経由でDINOv2をロード(初回のみダウンロード)
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval().to(self.device)
        self.dim = self.model.embed_dim  # vits14: 384

        self.transform = T.Compose(
            [
                T.ToTensor(),                      # [0,255] -> [0,1]
                T.Resize(
                    (INPUT_SIZE, INPUT_SIZE),
                    interpolation=T.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    @torch.no_grad()
    def embed(self, bw_image: np.ndarray) -> np.ndarray:
        """前処理済みのグレースケール/二値画像を受け取り、
        L2正規化済みベクトル(np.float32)を返す。
        """
        # グレースケール -> 3ch RGB (DINOv2は3ch入力)
        if bw_image.ndim == 2:
            rgb = np.stack([bw_image] * 3, axis=-1)
        else:
            rgb = bw_image
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        vec = self.model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        # L2正規化しておくと、内積 = コサイン類似度として扱える
        vec /= np.linalg.norm(vec) + 1e-8
        return vec

    @torch.no_grad()
    def embed_batch(self, bw_images: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """複数画像をまとめてベクトル化(一括登録の高速化用)。"""
        vecs = []
        for i in range(0, len(bw_images), batch_size):
            batch = bw_images[i : i + batch_size]
            tensors = torch.stack(
                [
                    self.transform(
                        np.stack([img] * 3, axis=-1) if img.ndim == 2 else img
                    )
                    for img in batch
                ]
            ).to(self.device)
            out = self.model(tensors).cpu().numpy().astype(np.float32)
            out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-8
            vecs.append(out)
        return np.vstack(vecs)


if __name__ == "__main__":
    # 動作確認: python embedder.py 図面ファイル
    import sys

    from preprocess import preprocess_for_embedding

    emb = DrawingEmbedder()
    vec = emb.embed(preprocess_for_embedding(sys.argv[1]))
    print(f"ベクトル化完了: 次元={len(vec)}, 先頭5要素={vec[:5]}")
