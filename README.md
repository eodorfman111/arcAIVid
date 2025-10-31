# arcAIVid  
**Automated Fish Detection & Counting for ARC Marine BRUV Footage**

---

## 🧭 Overview
`arcAIVid` processes underwater BRUV videos to automatically detect fish, identify the most active frames, and produce summary reports (CSV, XLSX, charts, and a PDF).  
It is built with **Ultralytics YOLO**, **Streamlit**, and **OpenCV**, and runs either:
- as a **stand-alone desktop app (EXE)**, or  
- directly from **Python** for development.

---

## 🐟 Features
- Detect fish in long (up to 60-minute) BRUV recordings.  
- Automatically divide each video into 10-minute sections.  
- Save top frames with bounding boxes for each section.  
- Output:
  - CSV and XLSX tables of detections
  - Plotly bar/density charts
  - A formatted PDF report
  - A single ZIP of all outputs
- CPU and GPU compatible (uses CUDA if available).

---


---

## ⚙️ Local Run (Development Mode)
Requires **Python 3.12 x64**.

```bash
cd arcAIVid_app
py -3.12 -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py

