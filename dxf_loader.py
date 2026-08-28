"""
dxf_loader.py - DXFファイルの読み込み・画像化・テキスト抽出モジュール

CADデータ(DXF)はスキャン図面と違い、線が座標データとして入っているため:
  - ノイズ・傾き・かすれが存在しない(前処理の大半が不要)
  - 寸法(DIMENSION)・文字(TEXT/MTEXT)をエンティティ型で確実に除去できる
  - 表題欄の品番・材質などをOCRなしでテキスト抽出できる場合がある

依存: pip install ezdxf matplotlib
DWG形式は直接読めないため、ODA File Converter等でDXFに変換してから使う。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.config import BackgroundPolicy, ColorPolicy, Configuration
from ezdxf.addons.drawing.matplotlib import MatplotlibBackend

import matplotlib

matplotlib.use("Agg")  # 画面表示なしのサーバ環境でも動くバックエンド
import matplotlib.pyplot as plt

# 形状の類似検索から除外するエンティティ型
# DIMENSION: 寸法 / TEXT・MTEXT: 注記文字 / LEADER系: 引出線 / HATCH: ハッチング
EXCLUDE_DXFTYPES = {
    "DIMENSION",
    "ARC_DIMENSION",
    "TEXT",
    "MTEXT",
    "LEADER",
    "MULTILEADER",
    "MLEADER",
}

# レイヤー名にこれらの語を含む場合も除外(社内CADルールに合わせて追加する)
EXCLUDE_LAYER_KEYWORDS = ("寸法", "注記", "図枠", "DIM", "TEXT", "FRAME", "TITLE")

RENDER_SIZE = 1600  # レンダリング画像の長辺ピクセル数(preprocess.MAX_SIDEと合わせる)


def _should_exclude(entity, exclude_annotations: bool) -> bool:
    """形状検索の邪魔になるエンティティかどうかを判定する。"""
    if not exclude_annotations:
        return False
    if entity.dxftype() in EXCLUDE_DXFTYPES:
        return True
    layer = (entity.dxf.layer or "").upper()
    return any(kw.upper() in layer for kw in EXCLUDE_LAYER_KEYWORDS)


def load_dxf_as_image(
    path: str | Path, exclude_annotations: bool = True, render_size: int = RENDER_SIZE
) -> np.ndarray:
    """DXFを読み込み、白背景・黒線のグレースケール画像(np.uint8)を返す。

    exclude_annotations=True なら寸法・文字・引出線を除いた「形状だけ」の
    画像を生成する。スキャン図面のsuppress_thin_linesと違い、確実に消える。
    """
    doc = ezdxf.readfile(str(path))
    msp = doc.modelspace()

    # 除外対象をフィルタするイテレータ
    entities = [e for e in msp if not _should_exclude(e, exclude_annotations)]
    if not entities:
        raise ValueError(f"描画対象のエンティティがありません: {path}")

    # 白背景・黒線でレンダリング(モデル空間の色設定に依存させない)
    config = Configuration(
        background_policy=BackgroundPolicy.WHITE,
        color_policy=ColorPolicy.BLACK,
    )

    fig = plt.figure(dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ctx = RenderContext(doc)
    backend = MatplotlibBackend(ax)
    Frontend(ctx, backend, config=config).draw_entities(entities)
    ax.set_axis_off()

    # 図形範囲にフィットさせ、長辺render_sizeで画像化
    ax.autoscale(True)
    ax.set_aspect("equal")
    fig.canvas.draw()
    w_in, h_in = fig.get_size_inches()
    scale = render_size / (max(w_in, h_in) * 100)
    fig.set_size_inches(w_in * scale, h_in * scale)
    fig.canvas.draw()

    # matplotlib描画結果 → numpyグレースケール
    buf = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    gray = np.dot(buf[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    return gray


def extract_dxf_text(path: str | Path, max_items: int = 100) -> list[str]:
    """DXF内のTEXT/MTEXTエンティティから文字列を抽出する。

    表題欄の品番・材質・板厚などがテキストで入っている場合、
    OCRなしで付帯情報を取得できる。抽出結果をLLMに渡して
    「品番・材質・板厚をJSONで返して」と構造化させる使い方を想定。
    """
    doc = ezdxf.readfile(str(path))
    texts: list[str] = []
    for e in doc.modelspace():
        t = e.dxftype()
        if t == "TEXT":
            s = e.dxf.text.strip()
        elif t == "MTEXT":
            s = e.plain_text().strip()
        else:
            continue
        if s:
            texts.append(s)
        if len(texts) >= max_items:
            break
    return texts


if __name__ == "__main__":
    # 動作確認: python dxf_loader.py 図面.dxf
    import sys

    import cv2

    img = load_dxf_as_image(sys.argv[1])
    cv2.imwrite("dxf_rendered_sample.png", img)
    print(f"レンダリング結果を dxf_rendered_sample.png に保存 (size={img.shape})")
    texts = extract_dxf_text(sys.argv[1])
    print(f"抽出テキスト {len(texts)}件: {texts[:10]}")
