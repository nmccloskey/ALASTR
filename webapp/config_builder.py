import streamlit as st
import yaml
import os
import sys
import tempfile
import zipfile
from io import BytesIO
from config_builder import build_config_ui

def add_src_to_sys_path():
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
add_src_to_sys_path()

from src.alastr.main import run_alastr_pipeline  # Adapt this import based on your structure

st.title("ALASTR Web App")

if "confirmed_config" not in st.session_state:
    st.session_state.confirmed_config = False

def zip_folder(folder_path):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zf.write(file_path, arcname)
    zip_buffer.seek(0)
    return zip_buffer

st.header("Step 1: Provide config and input files")

# Upload config or build it
config_file = st.file_uploader("Upload your config.yaml", type=["yaml", "yml"])
config = None

if config_file:
    st.session_state.confirmed_config = False  # reset if new file uploaded
    config = yaml.safe_load(config_file)
    st.success("✅ Config file uploaded")
else:
    with st.expander("No config uploaded? Build one here"):
        config = build_config_ui()
        if st.button("✅ Use this built config"):
            st.session_state.confirmed_config = True
            st.success("Built config confirmed.")

# Upload .cha files
cha_files = st.file_uploader("Upload .cha files", type=["cha"], accept_multiple_files=True)

if (config_file or st.session_state.confirmed_config) and cha_files:
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "input")
        output_dir = os.path.join(tmpdir, "output")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        config["input_dir"] = input_dir
        config["output_dir"] = output_dir

        # Save uploaded .cha files
        for file in cha_files:
            with open(os.path.join(input_dir, file.name), "wb") as f:
                f.write(file.read())

        if st.button("Run ALASTR Pipeline"):
            try:
                run_alastr_pipeline(config)  # You can expose a CLI-style `main(config)` or step-by-step logic here
                st.success("🎉 ALASTR pipeline completed!")

                zip_buffer = zip_folder(output_dir)
                st.download_button(
                    label="📦 Download Results ZIP",
                    data=zip_buffer,
                    file_name="alastr_output.zip",
                    mime="application/zip"
                )
            except Exception as e:
                st.error(f"Error running pipeline: {e}")
