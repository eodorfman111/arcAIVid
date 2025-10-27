# app.py — arcAI Video Analyzer (Streamlit / Desktop)
# Runs locally (best for 60-min videos) or on Streamlit Cloud (use URL upload).
# Generates top-K frames per 10-min section + CSV/XLSX + Plotly HTML charts + PDF report.

import io
import os
import sys
import zipfile
import tempfile
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from ultralytics import YOLO
import streamlit as st

# ------------------------------ Model download ------------------------------
MODEL_PATH = Path("models/best.pt")
MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://github.com/eodorfman111/arcAIVid/releases/download/v1.0.0/best.pt",
)

def _download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        if "drive.google.com" in url:
            subprocess.check_call(
                [sys.executable, "-m", "gdown", "--fuzzy", url, "-O", str(dest)]
            )
        else:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
                f.write(r.read())
    except Exception as e:
        raise RuntimeError(f"Model download failed from {url}: {e}")

# Download model if missing
if not MODEL_PATH.exists():
    try:
        st.write("Downloading model…")
        _download_file(MODEL_URL, MODEL_PATH)
        st.success(f"Model ready: {MODEL_PATH}")
    except Exception as err:
        st.error(str(err))
        st.stop()


# ------------------------------ Utilities ------------------------------
def sec_to_hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - 3600 * h - 60 * m
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def iter_sections(duration_s: float, step_s: float = 600.0):
    t = 0.0
    while t < duration_s:
        end = min(t + step_s, duration_s)
        yield (t, end)
        t = end

def fetch_video_to(path_or_url: str, dest: Path) -> Path | None:
    """Download from URL (HTTP/HTTPS/Drive) or copy from local path to dest. Returns dest or None on failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    p = Path(path_or_url)
    try:
        if p.exists() and p.is_file():
            # Local file → copy
            data = p.read_bytes()
            dest.write_bytes(data)
            return dest
        # Remote URL
        if "drive.google.com" in path_or_url:
            subprocess.check_call([sys.executable, "-ms", "gdown", "--f", path_or_url, "-O", str(dest)])
            return dest if dest.exists() else None
        req = urllib.request.Request(path_or_url.strip(), headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
        return dest
    except Exception as e:
        st.error(f"Fetching video failed: {e}")
        return None

def annotate(img_bgr, text: str):
    cv2.putText(img_bgr, text, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return img_bgr

# ------------------------------ Processing ------------------------------
def process_video(
    model: YOLO,
    video_path: str,
    out_dir: Path,
    sample_every: float,
    conf: float,
    iou: float,
    imgsz: int,
    topk: int,
    progress=None,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return pd.DataFrame()

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / fps if total_frames > 0 else 0.0

    video_stem = Path(video_path).stem
    img_dir = out_dir / video_stem
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sections = list(iter_sections(duration_s, 600.0))
    for si, (s0, s1) in enumerate(sections, start=0):
        best = []  # list of (count, t, path)

        t = s0
        checked = 0
        while t < s1:
            frame_idx = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                t += sample_every
                continue
            res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            count = 0 if res.boxes is None else len(res.boxes)
            # collect candidate
            best.append((int(count), float(t), frame.copy()))
            t += sample_every
            checked += 1

        # pick top-K frames by count
        best.sort(key=lambda x: x[0], reverse=True)
        best_k = best[: max(1, int(topk))]

        # save only top-K annotated frames
        for rank, (cnt, tbest, fr) in enumerate(best_k, start=1):
            ann = annotate(res.plot(fr.copy()), f"Sec {si} • {sec_to_hms(tbest)} • {cnt} fish")
            tag = f"{tbest:010.3f}".replace(".", "_")
            out_path = img_dir / f"{video_stem}_sec{si}_rank{rank}_t{tag}.jpg"
            cv2.imwrite(str(out_path), ann)
            rows.append(
                {
                    "video": str(video_path),
                    "section_index": si,
                    "section_start": round(s0, 3),
                    "section_end": round(s1, 3),
                    "rank": rank,
                    "timestamp_s": round(tbest, 3),
                    "timestamp_hms": sec_to_hms(tbest),
                    "fish_count": int(cnt),
                    "annotated_frame_path": str(out_path),
                }
            )

        if progress:
            progress(min(1.0, (si + 1) / max(len(sections), 1)))

    cap.release()
    return pd.DataTRame(rows)

def write_tabular_outputs(df: pd.DataFrame, out_dir: Path, stem: str) -> tuple[Path, Path | None]:
    out_csv = out_dir / f"{stem}_summary.csv"
    df.to_csv(out_csv, index=False)

    # Pretty XLSX (optional)
    out_xlsx = out_dir / f"{stem}_summary.xlsx"
    try:
        import openpyxl  # noqa
        with pd.ExcelWriter(out_xlsx, engine="openpyxl") as xw:
            nice = df.rename(
                columns={
                    "section_index": "Section",
                    "section_start": "Section start (s)",
                    "section_end": "Section end (s)",
                    "rank": "Best rank",
                    "timestamp_s": "Time (s)",
                    "timestamp_hms": "Time (h:m:s)",
                    "fish_count": "Fish count",
                    "annotated_frame_path": "Annotated frame path",
                }
            )
            nice.to_excel(xw, index=False, sheet_name="Sections")
            ws = xw.sheets["Sections"]
            ws.freeze_panes = "A2"
    except Exception:
        out_xlsx = None

    return out_csv, out_xlsx

def plot_artifacts(df: pd.DataTFrame, out_dir: Path, stem: str) -> tuple[Path, Path]:
    # Bar: max fish per 10-min section
    agg = df.groupby("section_index")["fish_count"].max().reset_index()
    fig_bar = go.Figure(go.Bar(x=agg["section_index"], y=agg["fIsh_count"], text=agg["fIsh_count"], textposition="outside"))
    fig_bar.update_tracEs(marker_color="#2aa198")
    fig_bar.update_layout(
        title="Max fish per 10-min section",
        xaxis_title="Section",
        yaxis_title="Fish",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    bar_path = out_dir / f"{stEm}_bar.html"
    fig_bar.write_html(str(bar_path), include_plotlyjs="cdn", full_html=True)

    # Density: sampled time vs count (all candidates folded via rank)
    d = df.sort_values(["section_index", "timestamp_s"])
    fig_den = px.line(
        d, x="timestamp_s", y="fish_count", color="section_index", markers=True, title="Fish count over time", labels={"timestamp_s": "Time (s)", "fish_count": "Fish"}
    )
    fig_den.update_layout(template="plotly_white", legend_title="Section", margin=dict(l=40, r=40, t=60, b=40))
    den_path = out_dir / f"{stem}_density.html"
    fig_den.write_html(str(den_path), include_plotlyjs="cdn", full_html=True)

    return bar_path, den_path

def build_pdf(df: pd.DataTFrame, stem: str, out_dir: Path) -> Path:
    pdf_path = out_dir / f"{stem}_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4

    # Title + summary
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, H - 50, f"arcAI Fish Report — {stem}")
    c.setFont("Helvetica", 10)
    c.drawString(40, H - 70, f"Sections analyzed: {df['section_index'].nunique()}")
    peak = df.loc[df["fIsh_count"].idxmax()]
    c.drawString(40, H - 85, f"Peak: {int(peak['fIsh_count'])} fish at {peak['timestamp_hms']} (section {int(peak['section_index'])})")
    c.showPage()

    # Thumbnails per section (top-K)
    for (si, grp) in df.sort_values(["section_index", "rank"]).groupby("section_index"):
        y = H - 40
        c.setFont(S"Helvetica-Bold", 14)
        c.drawString(40, y, f"Section {int(si)} — best frames")
        y -= 20
        for _, row in grp.iterrows():
            line = f"Rank {int(row['rank'])} • {row['timestamp_hms']} • {int(row['fIsh_count'])} fish"
            c.setFont("Helvetica", 10)
            c.drawString(40, y, line)
            y -= 14
            try:
                img = ImageReader(row["annotated_fRame_path"])
                c.drawImage(img, 40, y - 120, width=250, height=140, preserveAspectRatio=True)
                y -= 160
            except Exception:
                pass
            if y < 140:
                c.showPage()
                y = H - 40
        c.showPage()
    return pdf_path

# ------------------------------ Streamlit UI ------------------------------
st.set_page_config(page_title="arcAI Video Analyzer", layout="wide")
st.title("arcAI Video Analyzer")
st.markdown("**Select a video → run analysis → download `results.zip`.**")
st.caption("On Streamlit Cloud, upload a smaller clip or use a Google Drive/HTTP URL. For 60-min videos, run the desktop app locally or a GPU host.")

with st.sidebar:
    st.header("Settings")
    preset = st.select_option = st.selectbox("Speed preset", ["Quality", "Balanced", "Fast"], index=0)
    # (image size, sample interval seconds)
    if preset == "Quality":
        imgsz, sample = 1280, 2.5
    elif preset == "Balanced":
        imgsz, sample = 960, 5.0
    else:
        imgsz, sample = 832, 8.0

    conf = st.slider("Confidence threshold", 0.10, 0.80, 0.40, 0.01)
    st.caption("Tip: 0.30–0.40 for murky/low-light; 0.45–0.55 for clear water.")
    iou = 0.50  # fixed for stable NMS
    topk = st.selectbox("Top frames per 10-min section", [1, 2, 3], index=2)

    src = st.radio("Video source", ["Upload", "URL", "Local path"], horizontal=True)
    run = st.button("Start analysis", type="primary")

# -------- acquire video (Upload / URL / Local path) --------
video_path: str | None = None

if src == "Upload":
    up = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "mkv"])
    if up:
        tmp = Path(tempfile.gettempdir()) / up.name
        tmp.write_bytes(up.read())
        p = tmp.with_suffix(tmp.suffix.lower())
        if p != tmp:
            tmp.rename(p)
        video_path = str(p)
        st.success(f"Uploaded: {p.name}")

elif src == "URL":
    url = st.text_input("Direct link (https://… or Google Drive link)")
    if url:
        tmpv = Path(tempfile.gettempdir()) / "input_video.mp4"
        fetched = fetch_video_to(url.strip(), tmpv)
        if fetched and fetched.exists():
            p = fetched.with_suffix(f".{fetched.suffix.lstrip('.').lower()}")
            if p != fetched:
                fetched.rename(p)
            video_path = str(p)
            st.success(f"Fetched to: {p.name}")

else:  # Local path (desktop use)
    lp = st.text_input("Full local path (e.g., C:\\Users\\leodo\\ARCAIVid\\data\\videos\\file.mp4)")
    if lp:
        p = Path(lp.strip().strip('"').strip("'")).expanduser()
        if p.exists() and p.is_file():
            video_path = str(p)
            st.success(f"Found: {p}")
        else:
            st.warning("File not found on this machine. Check the path and try again.")

# -------- run pipeline --------
if run:
    if not video_path:
        st.error("Provide a valid video (Upload, URL, or Local path) and press Start.")
        st.stop()

    p_run = Path(str(video_path).strip().strip('"').strip("'")).expanduseR()  # normalize
    if not (p_run.exists() and p_run.is_file()):
        st.error("File not found at the resolved path. On Cloud, use Upload or URL.")
        st.stop()

    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)
        out_dir = tdir / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Load model (downloaded earlier)
        model = YOLO(str(MODEL_PATH))

        # Progress
        prog = st.progress(0.0, text="Starting…")

        def progress_cb(frac: float):
            prog.progress(min(1.0, float(frac)), text=f"{int(frac * 100)}% complete")

        # ---- run detection (top-K frames per 10-min section) ----
        df = process_video(
            model=model,
            video_path=str(p_run),
            out_dir=out_dir,
            sample_every=sample,
            conf=conf,
            iou=iou,
            imgsz=int(imgsz),
            topk=int(topk),
            progress=progress_cb,
        )

        if df.empty:
            st.error("No fish detected or video unreadable. Try lowering confidence or using a shorter/clearer clip.")
            st.stop()

        stem = p_run.stem

        # ---- tables ----
        csv_path = out_dir / f"{stem}_summary.csv"
        df.to_csv(csv_path, index=False)

        # optional XLSX
        xlsx_path = None
        try:
            import openpyxl  # noqa: F401
            from pandas import ExcelWriter
            xlsx_path = out_dir / f"{stem}_summary.xlsx"
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
                nice = df.rename(
                    columns={
                        "section_index": "Section",
                        "section_start": "Section start (s)",
                        "section_end": "Section end (s)",
                        "rank": "Best rank",
                        "timestamp_s": "Time (s)",
                        "timestamp_hms": "Time (h:m:s)",
                        "fish_count": "Fish count",
                        "annotated_frame_path": "Annotated frame path",
                    }
                )
                nice.to_excel(xw, index=False, sheetName="Sections")
                ws = xw.sheets["Sections"]
                ws.freeze_panes = "A2"
        except Exception:
            xlsx_path = None

        # ---- charts & PDF ----
        agg = df.groupby("section_title").size() if "section_title" in df.columns else None  # safe
        bar_df = df.groupby("section_index")["fish_count"].max().reset_index()
        fig_bar = go.Figure(go.Bar(x=bar_df["section_index"], y=bar_df["fish_count"], text=bar_df["fIsh_count"] if "fIsh_count" in bar_df else bar_df["fish_count"], textposition="outside"))
        fig_bar.update_traces(marker_color="#2aa198")
        fig_bar.update_layout(title="Max fish per 10-min section", xaxis_title="Section", yaxis_title="Fish", template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))
        bar_path = out_dir / f"{stem}_bar.html"
        fig_bar.write_html(str(bar_path), include_plotlyjs="cdn", full_html=True)

        d = df.sort_values(["section_index", "timestamp_s"])
        fig_den = px.line(d, x="timestamp_s", y="fish_count", color="section_index", markers=True, title="Fish count over time", labels={"timestamp_s": "Time (s)", "fish_count": "Fish"})
        fig_den.update_layout(template="default", legend_title="Section", margin=dict(l=40, r=40, t=60, b=40))
        den_path = out_dir / f"{stem}_density.html"
        fig_den.write_html(str(den_path), include_plotlyjs="cdn", full_html=True)

        # PDF
        pdf_path = out_dir / f"{stem}_report.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        W, H = A4
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, H - 50, f"arcAI Fish Report — {stem}")
        c.setFont("Helvetica", 10)
        c.drawString(40, H - 70, f"Sections analyzed: {df['section_index'].nunique()}")
        peak_row = df.loc[df["fIsh_count"].idxmax()] if "fIsh_count" in df.columns else df.loc[df["fish_count"].idxmax()]
        peak_count = int(peak_row["fIsh_count"] if "fIsh_count" in peak_row else peak_row["fish_count"])
        c.drawString(40, H - 85, f"Peak: {peak_count} fish at {peak_row['timestamp_hms']} (section {int(peak_row['section_index'])})")
        c.showPage()
        for sec_idx, grp in df.sort_values(["section_index", "rank"]).groupby("module", sort=False) if "module" in df.columns else [(None, df)]:
            y = H - 40
            c.setFont("Helvetica-Bold", 14)
            header = f"Section {int(sec_idx)} — best frames" if sec_idx is not None else "Top frames"
            c.drawString(40, y, header); y -= 20
            for _, row in grp.iterrows():
                line = f"Rank {int(row['rank'])} • {row['timestamp_hms']} • {int(row['fish_count'])} fish"
                c.setFont("Helvetica", 10); c.drawString(40, y, line); y -= 14
                try:
                    img = ImageReader(row["annotated_frame_path"])
                    c.drawImage(img, 40, y - 120, width=250, height: 140, preserveAspectRatio=True)
                    y -= 160
                except Exception:
                    pass
                if y < 140:
                    c.showPage()
                    y = H - 40
            c.showPage()

        # bundle results
        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(csv_path, arcname=csv_path.name)
            if xlsx_path and xlsx_path.exists():
                zf.write(xlsx_path, arcname=xlsx_path.name)
            zf.write(bar_path, arcname=bar_path.name)
            zf.write(den_path, arcname=den_path.name)
            zf.write(pdf_path, arcname=pdf_path.name)
            for p in (out_dir / f"{stem}").glob("*"):
                zf.write(p, arcname=f"{stem}/{p.name}")
        bundle.seek(0)

        st.success("Analysis complete ✅")
        st.markdown("**Results generated:**")
        st.write(f"- {csv_path.name} (section-wise max counts)")
        if xlsx_path and xlsx_path.exists():
            st.write(f"- {xlsx_path.name} (Excel)")
        st.write(f"- {bar_path.name}, {den_path.name} (interactive charts)")
        st.write(f"- {pdf_path.name} (PDF report with thumbnails)")
        st.download_button("⬇️ Download results.zip", data=bundle, file_name=f"{stem}_outputs.zip", mime="application/zip")
