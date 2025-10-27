# app.py — arcAI Video Analyzer (Streamlit Cloud ready)
import os, io, zipfile, tempfile, subprocess, sys, urllib.request, cv2, pandas as pd, numpy as np
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

# --------------------------------------------------
# Auto-download model from GitHub Release
# --------------------------------------------------
MODEL_PATH = Path("models/best.pt")
MODEL_URL = "https://github.com/eodorfman111/arcAIVid/releases/download/v1.0.0/best.pt"  # <-- replace if needed
if not MODEL_PATH.exists():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    st.info("Downloading model weights ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    st.success("Model downloaded.")

# --------------------------------------------------
# Utility functions
# --------------------------------------------------
def sec_to_hms(s): h=int(s//3600);m=int((s%3600)//60);ss=s-3600*h-60*m;return f"{h:02d}:{m:02d}:{ss:06.3f}"
def iter_sections(d, step=600):
    t=0.0
    while t<d: yield t,min(t+step,d);t+=step
def annotate(img, txt):
    cv2.putText(img,txt,(15,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA);return img

def fetch_video_from_url(url:str,dest:Path)->Path|None:
    dest.parent.mkdir(parents=True,exist_ok=True)
    try:
        if "drive.google.com" in url:
            subprocess.check_call([sys.executable,"-m","gdown","--fuzzy",url,"-O",str(dest)])
        else:
            urllib.request.urlretrieve(url,dest)
        return dest
    except Exception as e:
        st.error(f"Download failed: {e}")
        return None

# --------------------------------------------------
# Core processing
# --------------------------------------------------
def process_video(model,video_path,out_dir,sample_every,conf,iou,imgsz,topk=3,progress=None):
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():return pd.DataFrame()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur=frames/fps if frames>0 else 0
    stem=Path(video_path).stem;img_dir=Path(out_dir)/stem;img_dir.mkdir(parents=True,exist_ok=True)
    rows=[];secs=max(1,int(np.ceil(dur/600)));done=0;idx=0
    for s0,s1 in iter_sections(dur,600):
        scored=[];t=s0
        while t<s1:
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(t*fps)))
            ok,frame=cap.read()
            if not ok:t+=sample_every;continue
            r=model.predict(frame,conf=conf,iou=iou,imgsz=imgsz,verbose=False)[0]
            c=0 if r.boxes is None else len(r.boxes)
            scored.append((c,float(t)));t+=sample_every
        scored.sort(reverse=True,key=lambda x:x[0]);keep=scored[:max(1,int(topk))]
        for rank,(cnt,tt) in enumerate(keep,1):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(round(tt*fps)))
            ok,frame=cap.read()
            if not ok:continue
            r=model.predict(frame,conf=conf,iou=iou,imgsz=imgsz,verbose=False)[0]
            ann=annotate(r.plot(),f"Sec {idx} | {sec_to_hms(tt)} | {cnt} fish")
            tag=f"{tt:010.3f}".replace(".","_")
            out=img_dir/f"{stem}_sec{idx}_rank{rank}_t{tag}.jpg"
            cv2.imwrite(str(out),ann)
            rows.append(dict(section=idx,rank=rank,time_hms=sec_to_hms(tt),count=cnt,image_path=str(out)))
        idx+=1;done+=1
        if progress:progress(done/secs)
    cap.release();return pd.DataFrame(rows)

def plot_artifacts(df,out_dir,stem):
    g=df.groupby("section")["count"].max().reset_index()
    bar=go.Figure(go.Bar(x=g["section"],y=g["count"],text=g["count"],textposition="outside"))
    bar.update_layout(title="Max fish per 10-min section",xaxis_title="Section",yaxis_title="Fish",template="plotly_white")
    bar.write_html(str(Path(out_dir)/f"{stem}_bar.html"),include_plotlyjs="cdn",full_html=True)
    den=px.line(df,x="time_hms",y="count",color="section",title="Fish density (top-K)",template="plotly_white")
    den.write_html(str(Path(out_dir)/f"{stem}_density.html"),include_plotlyjs="cdn",full_html=True)

def build_pdf(df,stem,out_dir):
    pdf=Path(out_dir)/f"{stem}_report.pdf";c=canvas.Canvas(str(pdf),pagesize=A4);W,H=A4
    c.setFont("Helvetica-Bold",18);c.drawString(40,H-60,f"arcAI Report — {stem}")
    c.setFont("Helvetica",11);y=H-90
    c.drawString(40,y,f"Sections analyzed: {df['section'].nunique()}");y-=18
    peak=df.loc[df['count'].idxmax()];c.drawString(40,y,f"Peak: {int(peak['count'])} fish at {peak['time_hms']} (section {int(peak['section'])})")
    c.showPage()
    for _,r in df.sort_values(['section','rank']).iterrows():
        c.setFont("Helvetica-Bold",12)
        c.drawString(40,H-50,f"Section {int(r['section'])} | Rank {int(r['rank'])} | {r['time_hms']} | {int(r['count'])} fish")
        try:img=ImageReader(r['image_path']);c.drawImage(img,40,80,width=W-80,height=H-160,preserveAspectRatio=True,anchor="c")
        except:pass
        c.showPage()
    c.save();return str(pdf)

def zip_dir(path):
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,"w",zipfile.ZIP_DEFLATED)as z:
        for r,_,fs in os.walk(path):
            for f in fs:z.write(os.path.join(r,f),os.path.relpath(os.path.join(r,f),path))
    mem.seek(0);return mem

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="arcAI Video Analyzer",layout="wide")
st.title("arcAI Video Analyzer")
st.markdown("Upload or link a video → analyze fish counts → download results.zip")

with st.sidebar:
    st.header("Settings")
    preset=st.selectbox("Speed preset",["Quality","Balanced","Fast"])
    imgsz,sample=(1280,2.5) if preset=="Quality" else (960,5) if preset=="Balanced" else (832,8)
    conf=st.slider("Confidence",0.0,1.0,0.35,0.01)
    iou=st.slider("IoU",0.0,1.0,0.50,0.01)
    topk=st.selectbox("Top frames per section",[1,2,3],index=2)
    src=st.radio("Video source",["Upload","URL","Local path"],horizontal=True)
    run=st.button("Start",type="primary")

video_path=None
if src=="Upload":
    up=st.file_uploader("Upload video",type=["mp4","mov","avi","mkv"])
    if up:
        tmp=Path(tempfile.gettempdir())/up.name;tmp.write_bytes(up.read())
        video_path=str(tmp.with_suffix(tmp.suffix.lower()))
elif src=="URL":
    url=st.text_input("Paste video URL (HTTP/HTTPS or Google Drive)")
    if url:
        tmp=Path(tempfile.gettempdir())/"input_video.mp4"
        got=fetch_video_from_url(url,tmp)
        if got and got.exists():video_path=str(got)
else:
    lp=st.text_input("Full local path (e.g. C:\\Videos\\clip.mp4)")
    if lp and Path(lp).exists():video_path=lp

if run:
    if not video_path or not Path(video_path).exists():
        st.error("Provide a valid video.");st.stop()
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);out=tmp/"outputs";out.mkdir(parents=True,exist_ok=True)
        model=YOLO(str(MODEL_PATH))
        prog=st.progress(0.0,text="Initializing…")
        def cb(p):prog.progress(min(1.0,p),text=f"{int(p*100)}% done")
        df=process_video(model,video_path,out,sample,conf,iou,imgsz,int(topk),progress=cb)
        if df.empty:st.error("No frames processed.");st.stop()
        stem=Path(video_path).stem;csv=out/f"{stem}_summary.csv";df.to_csv(csv,index=False)
        plot_artifacts(df,out,stem);pdf=build_pdf(df,stem,out)
        z=zip_dir(out)
        st.download_button("Download results.zip",z,file_name=f"{stem}_outputs.zip",mime="application/zip")
        st.success("Analysis complete ✅")
