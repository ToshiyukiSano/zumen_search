"""
app.py - 類似図面検索の簡易Web UI (Streamlit)

起動:
    streamlit run app.py

ブラウザで図面をアップロードすると、登録済み図面から類似順に表示する。
現場の方にPoC評価をしてもらうための最小限のUI。
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from embedder import DrawingEmbedder
from indexer import DrawingIndex
from preprocess import make_thumbnail, preprocess_for_embedding

st.set_page_config(page_title="AI類似図面検索 PoC", layout="wide")


@st.cache_resource
def get_engine():
    """モデルとDBは初回のみロードし、以降は使い回す。"""
    embedder = DrawingEmbedder()
    index = DrawingIndex(dim=embedder.dim)
    return embedder, index


embedder, index = get_engine()

st.title("AI類似図面検索 (PoC)")
st.caption(f"登録済み図面: {index.count()} 件")

col_l, col_r = st.columns([1, 2])

with col_l:
    uploaded = st.file_uploader(
        "検索したい図面をアップロード", type=["pdf", "tif", "tiff", "png", "jpg", "jpeg", "dxf"]
    )
    top_k = st.slider("表示件数", 5, 30, 10)
    material = st.text_input("材質で絞り込み (任意)", placeholder="例: SPCC")
    customer = st.text_input("得意先で絞り込み (任意)")
    run = st.button("類似図面を検索", type="primary", disabled=uploaded is None)

if uploaded and run:
    # アップロードファイルを一時保存して処理
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = tmp.name

    with st.spinner("前処理・ベクトル化・検索中..."):
        try:
            bw = preprocess_for_embedding(tmp_path)
            vec = embedder.embed(bw)
        finally:
            # 図面は社外秘のため一時フォルダに残さない
            Path(tmp_path).unlink(missing_ok=True)
        filters = {}
        if material.strip():
            filters["material"] = material.strip()
        if customer.strip():
            filters["customer"] = customer.strip()
        results = index.search(vec, top_k=top_k, filters=filters or None)

    with col_l:
        st.subheader("検索元図面 (前処理後)")
        st.image(bw, use_container_width=True)

    with col_r:
        st.subheader(f"類似図面 上位{len(results)}件")
        if not results:
            st.warning("類似図面が見つかりませんでした。絞り込み条件を外すか、登録数を確認してください。")
        for rank, r in enumerate(results, 1):
            with st.container(border=True):
                c1, c2 = st.columns([1, 2])
                fp = r.get("file_path", "")
                with c1:
                    try:
                        c1.image(make_thumbnail(fp), use_container_width=True)
                    except Exception:
                        c1.text("(サムネイル生成不可)")
                with c2:
                    st.markdown(f"**{rank}位  類似度 {r['score']:.3f}**")
                    st.text(fp)
                    meta = {
                        k: v for k, v in r.items() if k not in ("score", "file_path")
                    }
                    if meta:
                        st.table(meta)
else:
    with col_r:
        st.info("左のフォームから図面をアップロードして検索してください。\n\n"
                "図面の登録はコマンドラインから:  `python register.py 図面フォルダ`")
