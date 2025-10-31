# run_app.py — no auto-open; fixed port 8501; no dev-mode conflict
import os, sys, tempfile
from pathlib import Path

CFG = """[global]
developmentMode = false
[server]
headless = true
address = "127.0.0.1"
port = 8501
[browser]
serverAddress = "localhost"
serverPort = 8501
"""

def main():
    # Block Streamlit from launching a browser
    os.environ["BROWSER"] = "none"

    # Neutralize conflicting env
    for k in ("STREAMLIT_SERVER_PORT","STREAMLIT_DEVELOPMENT_MODE",
              "STREAMLIT_GLOBAL_DEVELOPMENTMODE","STREAMLIT_RUN_ON_SAVE",
              "STREAMLIT_CONFIG_FILE"):
        os.environ.pop(k, None)

    # Our config
    cfg_path = Path(tempfile.gettempdir()) / "arcAIVid_streamlit.toml"
    cfg_path.write_text(CFG, encoding="utf-8")
    os.environ["STREAMLIT_CONFIG_FILE"] = str(cfg_path)

    base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent))
    app_path = str((base / "app.py").resolve())

    from streamlit.web import cli as stcli
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
