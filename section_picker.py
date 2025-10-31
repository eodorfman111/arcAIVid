# section_picker.py — CLI batch processor for arcAIVid
import argparse
from pathlib import Path
from ultralytics import YOLO
import pandas as pd
from app import process_video, MODEL_PATH  # reuse core logic

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="Folder with input videos")
    ap.add_argument("--model", default=str(MODEL_PATH), help="Path to best.pt")
    ap.add_argument("--out", default="outputs", help="Output folder")
    ap.add_argument("--sample-every", type=float, default=5.0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()

    model = YOLO(args.model)
    in_dir = Path(args.videos)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for vid in sorted(in_dir.glob("*")):
        if vid.suffix.lower() not in {".mp4",".mov",".mkv",".avi"}:
            continue
        df_top, fig_bar, fig_density, single_out = process_video(
            model=model,
            video_path=str(vid),
            out_dir=out_dir / vid.stem,
            sample_every=args.sample_every,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            topk=args.topk,
            progress=None
        )
        summaries.append(df_top)

    if summaries:
        all_sections = pd.concat(summaries, ignore_index=True)
        all_sections.to_csv(out_dir / "sections_summary.csv", index=False)

if __name__ == "__main__":
    main()
