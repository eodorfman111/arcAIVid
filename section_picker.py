# section_picker.py
"""
arcAI section picker
Splits a video into 10-minute sections, finds frames with most fish using YOLO,
and generates annotated images, CSV summary, charts, HTML dashboard, and PDF report.
"""

import argparse, os, cv2, json, io, base64, math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from jinja2 import Template

def sec_to_hms(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    ss = s - 3600*h - 60*m
    return f"{h:02d}:{m:02d}:{ss:06.3f}"

def iter_sections(duration_s, step_s=600):
    start = 0.0
    while start < duration_s:
        end = min(start + step_s, duration_s)
        yield (start, end)
        start = end

def draw_text(img, text):
    cv2.putText(img, text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return img

def process_video(model, vid_path, out_dir, sample_every, conf, iou, imgsz):
    cap = cv2.VideoCapture(str(vid_path))
    if not cap.isOpened():
        return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = frames / fps if frames > 0 else 0
    rows = []
    video_stem = Path(vid_path).stem
    img_out_dir = Path(out_dir) / video_stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    section_idx = 0
    for s0, s1 in iter_sections(duration_s, step_s=600):
        top_frames = []
        t = s0
        while t < s1:
            frame_idx = int(round(t * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                t += sample_every
                continue
            res = model.predict(frame, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
            count = 0 if res.boxes is None else len(res.boxes)
            annotated = res.plot()
            annotated = draw_text(annotated, f"{sec_to_hms(t)} | {count} fish")
            t_tag = f"{t:010.3f}".replace(".", "_")
            out_path = img_out_dir / f"{video_stem}_sec{section_idx}_t{t_tag}.jpg"
            cv2.imwrite(str(out_path), annotated)
            top_frames.append((count, t, str(out_path)))
            t += sample_every
        top_frames.sort(reverse=True, key=lambda x: x[0])
        for rank, (count, t, path) in enumerate(top_frames[:3]):
            rows.append({
                "video": str(vid_path),
                "section_index": section_idx,
                "rank": rank+1,
                "section_start_s": round(s0, 3),
                "section_end_s": round(s1, 3),
                "timestamp_s": round(t, 3),
                "timestamp_hms": sec_to_hms(t),
                "fish_count": count,
                "annotated_frame_path": path
            })
        section_idx += 1
    cap.release()
    return rows

def plot_counts(df, out_path):
    if df.empty: return
    df_group = df.groupby("section_index")["fish_count"].max().reset_index()
    plt.figure(figsize=(8,4))
    plt.bar(df_group["section_index"], df_group["fish_count"], color="steelblue")
    plt.xlabel("10-min Section")
    plt.ylabel("Max Fish Count")
    plt.title("Fish Count per Section")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_density(df, out_path):
    if df.empty: return
    plt.figure(figsize=(8,3))
    plt.plot(df["timestamp_s"], df["fish_count"], marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Fish Count")
    plt.title("Fish Density Over Time")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def build_html(df, video_stem, out_dir):
    html_tmpl = """
    <html><head><title>{{video_stem}} summary</title>
    <style>
    body{font-family:Arial;background:#f4f4f4;padding:20px;}
    table{border-collapse:collapse;width:100%;background:white;}
    th,td{border:1px solid #ccc;padding:6px;text-align:left;}
    img{max-width:200px;}
    </style></head><body>
    <h1>{{video_stem}}</h1>
    <table><tr><th>Section</th><th>Rank</th><th>Time</th><th>Count</th><th>Frame</th></tr>
    {% for _,r in df.iterrows() %}
    <tr>
      <td>{{r.section_index}}</td><td>{{r.rank}}</td><td>{{r.timestamp_hms}}</td>
      <td>{{r.fish_count}}</td>
      <td><img src="{{r.annotated_frame_path}}" /></td>
    </tr>
    {% endfor %}
    </table></body></html>"""
    html = Template(html_tmpl).render(video_stem=video_stem, df=df)
    html_path = Path(out_dir)/f"{video_stem}_dashboard.html"
    with open(html_path,"w",encoding="utf-8") as f: f.write(html)

def build_pdf(df, video_stem, out_dir):
    pdf_path = Path(out_dir)/f"{video_stem}_report.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    c.setFont("Helvetica-Bold",16)
    c.drawString(40,height-50,f"arcAI Fish Report - {video_stem}")
    c.setFont("Helvetica",10)
    y = height-80
    for _,r in df.iterrows():
        c.drawString(40,y,f"Section {r.section_index} Rank {r.rank} | {r.timestamp_hms} | {r.fish_count} fish")
        try:
            img = ImageReader(r.annotated_frame_path)
            c.drawImage(img,40,y-120,width=250,height=140,preserveAspectRatio=True)
            y -= 160
        except:
            y -= 20
        if y<120:
            c.showPage()
            y = height-80
            c.setFont("Helvetica",10)
    c.save()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="outputs/sections")
    ap.add_argument("--sample-every", type=float, default=2.5)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.50)
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    all_rows = []
    for ext in ("*.mp4","*.mov","*.avi","*.mkv"):
        for vp in Path(args.videos).glob(ext):
            print(f"Processing {vp} ...")
            rows = process_video(model, vp, args.out, args.sample_every, args.conf, args.iou, args.imgsz)
            if not rows: continue
            df = pd.DataFrame(rows)
            video_stem = Path(vp).stem
            csv_path = Path(args.out)/f"{video_stem}_summary.csv"
            df.to_csv(csv_path,index=False)
            plot_counts(df, Path(args.out)/f"{video_stem}_bar.png")
            plot_density(df, Path(args.out)/f"{video_stem}_density.png")
            build_html(df, video_stem, args.out)
            build_pdf(df, video_stem, args.out)
            all_rows.extend(rows)

    df_all = pd.DataFrame(all_rows)
    csv_all = Path(args.out)/"sections_summary.csv"
    df_all.to_csv(csv_all,index=False)
    print(f"Wrote {csv_all} with {len(df_all)} rows.")

if __name__ == "__main__":
    main()
