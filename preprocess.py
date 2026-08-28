"""
preprocess.py - 図面の前処理モジュール

PDF/TIFF/画像/DXFファイルを読み込み、類似検索に適した画像に変換する。
処理内容:
  1. PDFの画像化 (PyMuPDF, 300dpi相当) / DXFのレンダリング (dxf_loader)
  2. グレースケール化・二値化 (大津の方法)
  3. 傾き補正 (デスキュー) ※スキャン系のみ
  4. 寸法線・細線の軽減 (モルフォロジー処理) ※スキャン系のみ
     DXFは寸法・注記をエンティティ型で確実に除去済みのためスキップ
  5. 表題欄領域の切り出し (右下領域, AI-OCR用)
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np

# PDF読み込みに使用 (pip install pymupdf)
import fitz  # PyMuPDF

TARGET_DPI = 300          # PDF画像化の解像度
MAX_SIDE = 1600           # 埋め込み前の最大辺サイズ(処理速度と精度のバランス)
TITLE_BLOCK_RATIO = 0.30  # 表題欄とみなす右下領域の比率(幅・高さの30%)


def load_image(path: str | Path) -> np.ndarray:
    """PDF/TIFF/PNG/JPG を読み込み、グレースケール画像(np.ndarray)を返す。

    PDFは1ページ目のみ対象(図面は通常1ページ1図面のため)。
    複数ページPDFを扱う場合は register.py 側でページ分割する。
    """
    path = Path(path)
    if path.suffix.lower() == ".dxf":
        # CADデータ: dxf_loaderで寸法・注記を除外してレンダリング
        from dxf_loader import load_dxf_as_image

        return load_dxf_as_image(path)
    if path.suffix.lower() == ".pdf":
        doc = fitz.open(path)
        page = doc[0]
        zoom = TARGET_DPI / 72  # PDFの標準解像度は72dpi
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        doc.close()
        return img
    # TIFF/PNG/JPG等 (日本語パス対応のため np.fromfile 経由で読む)
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"画像を読み込めません: {path}")
    return img


def binarize(gray: np.ndarray) -> np.ndarray:
    """大津の方法で二値化。線=黒(0), 背景=白(255) に統一する。"""
    # 軽いノイズ除去
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 黒背景の図面(反転スキャン)なら反転して白背景に揃える
    if np.mean(bw) < 127:
        bw = cv2.bitwise_not(bw)
    return bw


def deskew(bw: np.ndarray, max_angle: float = 5.0) -> np.ndarray:
    """スキャン時の軽微な傾きを補正する(±max_angle度まで)。

    最小外接矩形の角度から推定する簡易方式。
    傾きが大きい図面が多い場合はHough変換ベースに差し替える。
    """
    inverted = cv2.bitwise_not(bw)  # 線を白にして座標を取る
    coords = np.column_stack(np.where(inverted > 0))
    if len(coords) < 100:
        return bw
    angle = cv2.minAreaRect(coords)[-1]
    # minAreaRect の角度規約はOpenCVバージョンで異なるため ±45°に折り畳む
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90
    angle = -angle  # coords が (y,x) 順のため符号を反転すると補正方向になる
    if abs(angle) < 0.1 or abs(angle) > max_angle:
        return bw
    h, w = bw.shape
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        bw, m, (w, h), flags=cv2.INTER_NEAREST, borderValue=255
    )


def suppress_thin_lines(bw: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """寸法線・引出線などの細線を軽減する。

    モルフォロジーのオープニング(収縮→膨張)で、太い外形線を残しつつ
    細い線を消す。kernel_size を大きくするほど強く消えるが、
    やり過ぎると外形線まで欠けるため 2〜3 で現場データを見て調整する。
    """
    inverted = cv2.bitwise_not(bw)  # 線を白に
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size + 1, kernel_size + 1)
    )
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(opened)


def crop_title_block(bw: np.ndarray) -> np.ndarray:
    """表題欄(右下領域)を切り出す。AI-OCRへの入力用。

    JIS図面枠では表題欄は右下が標準。自社図面の様式が異なる場合は
    TITLE_BLOCK_RATIO や切り出し位置を調整する。
    """
    h, w = bw.shape
    y0 = int(h * (1 - TITLE_BLOCK_RATIO))
    x0 = int(w * (1 - TITLE_BLOCK_RATIO))
    return bw[y0:, x0:]


def resize_keep_aspect(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    """長辺が max_side になるよう縦横比を保って縮小する。"""
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
    )


def preprocess_for_embedding(path: str | Path, suppress_dims: bool = True) -> np.ndarray:
    """埋め込み(ベクトル化)用の前処理を一括実行し、白背景の二値画像を返す。

    DXFはレンダリング時点で寸法・注記が除去済みかつ傾きゼロのため、
    デスキューと細線除去はスキップする(細線除去をかけると
    レンダリングされた細い外形線まで消えてしまう)。
    """
    is_dxf = Path(path).suffix.lower() == ".dxf"
    gray = load_image(path)
    bw = binarize(gray)
    if not is_dxf:
        bw = deskew(bw)
        if suppress_dims:
            bw = suppress_thin_lines(bw)
    bw = resize_keep_aspect(bw)
    return bw


def make_thumbnail(path: str | Path, max_side: int = 480) -> bytes:
    """一覧表示用のサムネイル(PNGバイト列)を生成する。"""
    gray = load_image(path)
    thumb = resize_keep_aspect(gray, max_side)
    ok, buf = cv2.imencode(".png", thumb)
    if not ok:
        raise RuntimeError("サムネイル生成に失敗しました")
    return buf.tobytes()


if __name__ == "__main__":
    # 動作確認: python preprocess.py 図面ファイル
    import sys

    out = preprocess_for_embedding(sys.argv[1])
    cv2.imwrite("preprocessed_sample.png", out)
    print(f"前処理結果を preprocessed_sample.png に保存しました (size={out.shape})")
