# app.py — Streamlit UI for arcAIVid
import os
import io
import math
import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ultralytics import YOLO
import cv2
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# ---------- CONFIG ----------
APP_NAME = "arcAIVid"
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)

# Optional: set to a GitHub Release asset URL to auto-download best.pt if missing.
MODEL_URL = ""  # e.g. "https://github.com/USER/REPO/releases/download/v1.0.0/best.pt"
MODEL_PATH = MODELS_DIR / "best.pt"

# ---------- UTIL ----------
def _seconds_to_hms(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))

def _ensure_model():
    if MODEL_PATH.exists():
        return
    if not MODEL_URL:
        raise FileNotFoundError(f"Missing model weights: {MODEL_PATH} and no MODEL_URL set.")
    try:
        import gdown
    except Exception as e:
        raise RuntimeError("gdown is required to download MODEL_URL. Add it to requirements.txt") from e
    st.info("Downloading model weights...")
    gdown.download(MODEL_URL, str(MODEL_PATH), quiet=False, fuzzy=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Model download failed.")

def _annotate_and_save(model, frame_bgr, save_path, conf, iou, imgsz):
    # Run a second pass to get plotted frame
    results = model.predict(source=frame_bgr[..., ::-1],  # RGB
                            conf=conf, iou=iou, imgsz=imgsz, verbose=False)
    plotted = results[0].plot()  # BGR np.array
    cv2.imwrite(str(save_path), plotted)

def _write_pdf_report(pdf_path, title, df_top, sections_count, peak_info, avg_max):
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4
    margin = 36

    # Title page
    c.setFont("Helvetica-Bold", 18)
    c.drawString(margin, H - margin - 20, title)
    c.setFont("Helvetica", 12)
    y = H - margin - 60
    c.drawString(margin, y, f"Sections analyzed: {sections_count}")
    y -= 18
    c.drawString(margin, y, f"Peak fish count: {peak_info['fish_count']} at {peak_info['timestamp_hms']} (sec {int(peak_info['timestamp_s'])})")
    y -= 18
    c.drawString(margin, y, f"Average of section maxima: {avg_max:.2f}")
    c.showPage()

    # Section pages with thumbnails
    if df_top.empty:
        c.save()
        return

    # Group by section_index, sorted
    for sec_idx, group in df_top.sort_values(["section_index", "rank"]).groupby("section_index"):
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, H - margin - 20, f"Section {sec_idx}")
        y = H - margin - 60
        x = margin
        thumb_w = 220
        thumb_h = 124
        per_row = 2
        count_in_row = 0

        for _, row in group.iterrows():
            img_path = row.get("annotated_frame_path", "")
            if img_path and Path(img_path).exists():
                try:
                    c.drawImage(ImageReader(img_path), x, y - thumb_h, width=thumb_w, height=thumb_h, preserveAspectRatio=True, mask='auto')
                except Exception:
                    pass
            c.setFont("Helvetica", 10)
            caption = f"Rank {int(row['rank'])} | {row['fish_count']} fish | {row['timestamp_hms']}"
            c.drawString(x, y - thumb_h - 12, caption)

            count_in_row += 1
            if count_in_row == per_row:
                count_in_row = 0
                x = margin
                y -= (thumb_h + 36)
                if y < margin + 160:
                    c.showPage()
                    c.setFont("Helvetica-Bold", 14)
                    c.drawString(margin, H - margin - 20, f"Section {sec_idx} (cont.)")
                    y = H - margin - 60
            else:
                x += (thumb_w + 36)
        c.showPage()
    c.save()

def _zip_dir(folder: Path, zip_path: Path):
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in folder.rglob("*"):
            z.write(p, p.relative_to(folder))

# ---------- CORE PIPELINE ----------
def process_video(model, video_path, out_dir, sample_every, conf, iou, imgsz, topk, progress=None) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    video, section_index, rank, section_start_s, section_end_s,
    timestamp_s, timestamp_hms, fish_count, annotated_frame_path
    """
    out_dir = Path(out_dir)
    frames_dir = out_dir / "frames"
    charts_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_s = frame_count / fps if frame_count > 0 else 0

    # If duration cannot be derived, estimate from frame-by-frame seek.
    if duration_s <= 0:
        # fall back: probe by jumping large offsets until fails
        # but most BRUV files have metadata; keep it simple
        duration_s = 60 * 60  # assume 60 min if unknown

    section_len = 600.0  # 10 minutes
    num_sections = max(1, math.ceil(duration_s / section_len))

    # Sample timestamps
    timestamps = []
    t = 0.0
    while t < duration_s:
        timestamps.append(t)
        t += float(sample_every)

    # Predict counts
    rows = []
    total = len(timestamps)
    for idx, ts in enumerate(timestamps):
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue

        # Run YOLO inference on a single frame
        results = model.predict(source=frame[..., ::-1], conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        fish_count = int(len(results[0].boxes)) if results and len(results) else 0

        sec_idx = int(ts // section_len)
        sec_start = sec_idx * section_len
        sec_end = min((sec_idx + 1) * section_len, duration_s)

        rows.append({
            "video": Path(video_path).name,
            "section_index": sec_idx,
            "section_start_s": sec_start,
            "section_end_s": sec_end,
            "timestamp_s": ts,
            "timestamp_hms": _seconds_to_hms(ts),
            "fish_count": fish_count
        })

        if progress:
            progress((idx + 1) / total)

    cap.release()

    if not rows:
        return pd.DataFrame(columns=[
            "video","section_index","rank","section_start_s","section_end_s",
            "timestamp_s","timestamp_hms","fish_count","annotated_frame_path"
        ])

    df = pd.DataFrame(rows)

    # Select top K per section
    df_top = (
        df.sort_values(["section_index", "fish_count", "timestamp_s"], ascending=[True, False, True])
          .groupby("section_index", as_index=False)
          .head(int(topk))
          .copy()
    )

    # Annotate and save frames for the chosen timestamps
    cap = cv2.VideoCapture(str(video_path))
    for i, r in df_top.iterrows():
        ts = float(r["timestamp_s"])
        cap.set(cv2.CAP_PROP_POS_MSEC, ts * 1000.0)
        ok, frame = cap.read()
        if not ok:
            df_top.loc[i, "annotated_frame_path"] = ""
            continue

        save_name = f"sec{int(r['section_index']):02d}_rank??_t{int(ts)}s.jpg"
        # Fill rank later after ranking
        tmp_path = frames_dir / save_name
        _annotate_and_save(model, frame, tmp_path, conf, iou, imgsz)
        df_top.loc[i, "annotated_frame_path"] = str(tmp_path)

    cap.release()

    # Assign ranks within each section by fish_count desc then time asc
    df_top = (df_top
              .sort_values(["section_index","fish_count","timestamp_s"], ascending=[True, False, True])
              .groupby("section_index", as_index=False)
              .apply(lambda g: g.assign(rank=range(1, len(g)+1)))
              .reset_index(drop=True))

    # Rename files to include final rank
    for i, r in df_top.iterrows():
        p = Path(r["annotated_frame_path"])
        if not p.exists():
            continue
        new_name = f"sec{int(r['section_index']):02d}_rank{int(r['rank']):02d}_t{int(r['timestamp_s'])}s.jpg"
        new_path = p.with_name(new_name)
        try:
            p.rename(new_path)
        except Exception:
            new_path = p  # keep old if rename fails
        df_top.loc[i, "annotated_frame_path"] = str(new_path)

    # Build charts
    # 1) Bar of max per section
    max_per_section = df.groupby("section_index", as_index=False)["fish_count"].max()
    fig_bar = px.bar(max_per_section, x="section_index", y="fish_count",
                     title="Max fish per 10-min section")
    # 2) Density (histogram) of fish_count over all samples
    fig_density = px.histogram(df, x="fish_count", nbins=30, marginal="rug",
                               title="Fish count distribution across samples")

    # Save data and charts
    summary_csv = out_dir / f"{Path(video_path).stem}_summary.csv"
    summary_xlsx = out_dir / f"{Path(video_path).stem}_summary.xlsx"
    bar_html = out_dir / f"{Path(video_path).stem}_bar.html"
    density_html = out_dir / f"{Path(video_path).stem}_density.html"
    pdf_path = out_dir / f"{Path(video_path).stem}_report.pdf"

    with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="samples")
        df_top.to_excel(xw, index=False, sheet_name="top_frames")
        max_per_section.to_excel(xw, index=False, sheet_name="section_maxima")

    df_top.to_csv(summary_csv, index=False)
    fig_bar.write_html(str(bar_html), include_plotlyjs="cdn")
    fig_density.write_html(str(density_html), include_plotlyjs="cdn")

    # PDF report
    peak_row = max_per_section.sort_values("fish_count", ascending=False).head(1)
    if not peak_row.empty:
        peak_sec = int(peak_row.iloc[0]["section_index"])
        peak_time_row = df[df["section_index"] == peak_sec].sort_values("fish_count", ascending=False).head(1).iloc[0]
        peak_info = {"fish_count": int(peak_row.iloc[0]["fish_count"]),
                     "timestamp_s": float(peak_time_row["timestamp_s"]),
                     "timestamp_hms": peak_time_row["timestamp_hms"]}
    else:
        peak_info = {"fish_count": 0, "timestamp_s": 0, "timestamp_hms": "0:00:00"}

    avg_max = float(max_per_section["fish_count"].mean()) if not max_per_section.empty else 0.0
    _write_pdf_report(pdf_path,
                      title=f"{APP_NAME} report — {Path(video_path).name}",
                      df_top=df_top,
                      sections_count=num_sections,
                      peak_info=peak_info,
                      avg_max=avg_max)

    # Return final table with rank and annotated paths
    # include section start/end and video name
    df_top = df_top[[
        "video","section_index","section_start_s","section_end_s",
        "rank","timestamp_s","timestamp_hms","fish_count","annotated_frame_path"
    ]].sort_values(["section_index","rank"])
    return df_top, fig_bar, fig_density, out_dir

# ---------- UI ----------
st.set_page_config(page_title=APP_NAME, layout="wide")
st.title(APP_NAME)
st.caption("ARC Marine fish detection and top-frame picker")

# Presets
preset = st.sidebar.selectbox(
    "Speed preset",
    ["Quality (1280 / 2.5 s)", "Balanced (960 / 5 s)", "Fast (832 / 8 s)"],
    index=1
)
if preset.startswith("Quality"):
    imgsz, sample_every = 1280, 2.5
elif preset.startswith("Fast"):
    imgsz, sample_every = 832, 8
else:
    imgsz, sample_every = 960, 5

conf = st.sidebar.slider("Confidence", 0.10, 0.90, 0.40, 0.01)
iou = st.sidebar.slider("IoU", 0.10, 0.90, 0.50, 0.01)
topk = st.sidebar.slider("Top frames per 10-min section", 1, 5, 3, 1)

mode = st.radio("Source", ["Upload", "URL", "Local Path"], horizontal=True)
uploaded_file = None
video_path = None
tmp_dir = Path(tempfile.mkdtemp(prefix="arcAIVid_"))

try:
    if mode == "Upload":
        f = st.file_uploader("Upload MP4/MOV (<= ~500 MB)", type=["mp4","mov","mkv","avi"])
        if f:
            p = tmp_dir / f.name
            with open(p, "wb") as out:
                out.write(f.read())
            video_path = str(p)
    elif mode == "URL":
        url = st.text_input("Direct URL or Google Drive share link")
        if url:
            try:
                import gdown
                st.info("Downloading video...")
                local_path = tmp_dir / "input.mp4"
                gdown.download(url, str(local_path), fuzzy=True, quiet=False)
                if local_path.exists() and local_path.stat().st_size > 0:
                    video_path = str(local_path)
                else:
                    st.error("Download failed or empty file.")
            except Exception as e:
                st.error(f"URL download error: {e}")
    else:
        video_path = st.text_input("Local path (e.g., D:\\BRUV\\clip001.mp4)").strip() or None
        if video_path and not Path(video_path).exists():
            st.warning("File not found.")

    col_run, col_model = st.columns([1,1])
    with col_model:
        st.write("Model file")
        st.code(str(MODEL_PATH), language="text")

    if st.button("Start analysis", disabled=not video_path):
        try:
            _ensure_model()
            st.info("Loading model...")
            model = YOLO(str(MODEL_PATH))

            prog = st.progress(0.0)
            df_top, fig_bar, fig_density, out_dir = process_video(
                model=model,
                video_path=video_path,
                out_dir=OUTPUTS_DIR / Path(video_path).stem,
                sample_every=sample_every,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                topk=topk,
                progress=lambda frac: prog.progress(min(0.999, float(frac)))
            )
            prog.progress(1.0)

            st.subheader("Top frames per section")
            st.dataframe(df_top, use_container_width=True)

            st.subheader("Charts")
            st.plotly_chart(fig_bar, use_container_width=True)
            st.plotly_chart(fig_density, use_container_width=True)

            # Build zip
            zip_path = out_dir.with_suffix(".zip")
            _zip_dir(out_dir, zip_path)

            # Offer downloads
            c1, c2, c3, c4, c5 = st.columns(5)
            files = [
                out_dir / f"{Path(video_path).stem}_summary.csv",
                out_dir / f"{Path(video_path).stem}_summary.xlsx",
                out_dir / f"{Path(video_path).stem}_bar.html",
                out_dir / f"{Path(video_path).stem}_density.html",
                out_dir / f"{Path(video_path).stem}_report.pdf",
            ]
            for col, fp in zip([c1,c2,c3,c4,c5], files):
                if fp.exists():
                    with open(fp, "rb") as fh:
                        col.download_button(f"Download {fp.name}", fh.read(), file_name=fp.name, mime="application/octet-stream")

            if zip_path.exists():
                with open(zip_path, "rb") as fh:
                    st.download_button("Download results.zip", fh.read(), file_name=zip_path.name, mime="application/zip")

            st.success("Done.")
            st.caption(f"Outputs folder: {out_dir}")

        except Exception as e:
            st.error(f"Error: {e}")

finally:
    # Do not remove tmp_dir so users can re-run in session. Comment out to keep.
    pass
