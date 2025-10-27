# section_picker.py — CLI batch processor
import argparse, os, cv2, pandas as pd, numpy as np
from pathlib import Path
from ultralytics import YOLO
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

def sec_to_hms(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); ss = s - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{ss:06.3f}"

def iter_sections(duration_s: float, step_s: int = 600):
    t = 0.0
    while t < duration_s:
        yield t, min(t + step_s, duration_s)
        t += step_s

def annotate(frame_bgr, text: str):
    img = frame_bgr.copy()
    cv2.putText(img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return img

def process_video(model, vp: Path, out_dir: Path, sample_every, conf, iou, imgsz, topk=3):
    cap = cv2.VideoCapture(str(vp))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frames / fps if frames > 0 else 0

    video_stem = vp.stem
    img_dir = out_dir / video_stem
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    sec_idx = 0
    while True:
        secs = list(iter_sections(duration_s, 600))
        if sec_idx >= len(secs): break
        s0, s1 = secs[sec_idx]

        # score only
        scored = []
        t = s0
        while t < s1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, frame = cap.read()
            if not ok:
                t += sample_every
                continue
            r = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            cnt = 0 if r.boxes is None else len(r.boxes)
            scored.append((cnt, float(t)))
            t += sample_every

        scored.sort(reverse=True, key=lambda x: x[0])
        keep = scored[:max(1, int(topk))]

        for rank, (cnt, tt) in enumerate(keep, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(tt * fps)))
            ok, frame = cap.read()
            if not ok: continue
            r = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            ann = annotate(r.plot(), f"Sec {sec_idx} | {sec_to_hms(tt)} | {cnt} fish")
            t_tag = f"{tt:010.3f}".replace(".", "_")
            out_jpg = img_dir / f"{video_stem}_sec{sec_idx}_rank{rank}_t{t_tag}.jpg"
            cv2.imwrite(str(out_jpg), ann)

            rows.append({
                "video": str(vp),
                "section": sec_idx,
                "rank": rank,
                "start_hms": sec_to_hms(s0),
                "end_hms": sec_to_hms(s1),
                "time_hms": sec_to_hms(tt),
                "time_s": round(tt, 3),
                "count": int(cnt),
                "image_path": str(out_jpg)
            })

        sec_idx += 1

    cap.release()
    return rows

def plot_artifacts(df: pd.DataFrame, out_dir: Path, stem: str):
    g = df.groupby("section")["count"].max().reset_index()
    fig_bar = go.Figure(go.Bar(x=g["section"], y=g["count"], text=g["count"], textposition="outside"))
    fig_bar.update_layout(title="Max fish per 10-min section", xaxis_title="Section", yaxis_title="Fish",
                          template="plotly_white", uniformtext_minsize=10, uniformtext_mode="hide",
                          margin=dict(l=40,r=20,t=60,b=40))
    fig_bar.update_yaxes(showgrid=True, gridcolor="#eee")
    fig_bar.write_html(str(out_dir / f"{stem}_plotly_bar.html"), include_plotlyjs="cdn", full_html=True)

    d2 = df.sort_values(["section","time_s"])
    fig_den = px.line(d2, x="time_s", y="count", color="section",
                      title="Fish density over sampled frames (top-K)",
                      template="plotly_white")
    fig_den.update_layout(legend_title_text="Section", margin=dict(l=40,r=20,t=60,b=40))
    fig_den.write_html(str(out_dir / f"{stem}_plotly_density.html"), include_plotlyjs="cdn", full_html=True)

def build_pdf(df: pd.DataFrame, stem: str, out_dir: Path) -> str:
    pdf_path = Path(out_dir) / f"{stem}_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    W, H = A4

    c.setFont("Helvetica-Bold", 18); c.drawString(40, H-60, f"arcAI Fish Report — {stem}")
    c.setFont("Helvetica", 11)
    secs = df["section"].nunique()
    max_row = df.loc[df["count"].idxmax()]
    avg_per_sec = float(df.groupby("section")["count"].max().mean().round(2))
    y = H-90
    for ln in [
        f"Sections analyzed: {secs}",
        f"Peak count: {int(max_row['count'])} fish at {max_row['time_hms']} (section {int(max_row['section'])})",
        f"Average max per section: {avg_per_sec}",
    ]:
        c.drawString(40, y, ln); y -= 18
    y -= 10
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, "Section"); c.drawString(100, y, "Rank"); c.drawString(140, y, "Time"); c.drawString(220, y, "Count")
    c.setFont("Helvetica", 10); y -= 14
    for _, r in df.sort_values(["section","rank"]).iterrows():
        c.drawString(40, y, str(int(r["section"]))); c.drawString(100, y, str(int(r["rank"])))
        c.drawString(140, y, r["time_hms"]); c.drawString(220, y, str(int(r["count"])))
        y -= 14
        if y < 120:
            c.showPage(); c.setFont("Helvetica", 10); y = H-60
    c.showPage()

    for _, r in df.sort_values(["section","rank"]).iterrows():
        c.setFont("Helvetica-Bold", 12)
        c.drawString(40, H-50, f"Section {int(r['section'])} | Rank {int(r['rank'])} | {r['time_hms']} | {int(r['count'])} fish")
        try:
            img = ImageReader(r["image_path"])
            c.drawImage(img, 40, 80, width=W-80, height=H-160, preserveAspectRatio=True, anchor="c")
        except Exception:
            c.setFont("Helvetica", 10); c.drawString(40, H-80, "Image not available.")
        c.showPage()

    c.save()
    return str(pdf_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="outputs/sections")
    ap.add_argument("--sample-every", type=float, default=2.5)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)

    all_rows = []
    for ext in ("*.mp4","*.mov","*.avi","*.mkv"):
        for vp in Path(args.videos).glob(ext):
            print(f"Processing {vp} ...")
            rows = process_video(model, vp, out_dir, args.sample_every, args.conf, args.iou, args.imgsz, topk=args.topk)
            if not rows: continue
            df = pd.DataFrame(rows)
            video_stem = Path(vp).stem
            cols = ["video","section","rank","start_hms","end_hms","time_hms","time_s","count","image_path"]
            df = df[cols]
            csv_path = out_dir / f"{video_stem}_summary.csv"
            df.to_csv(csv_path, index=False)
            plot_artifacts(df, out_dir, video_stem)
            build_pdf(df, video_stem, out_dir)
            all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows)
    csv_all = out_dir / "sections_summary.csv"
    df_all.to_csv(csv_all, index=False)
    print(f"Wrote {csv_all} with {len(df_all)} rows.")

if __name__ == "__main__":
    main()
