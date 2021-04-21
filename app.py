import streamlit as st
import importlib

files = {
    "---Please select---": "", 
    "Week 2": "Week2"
}

st.set_page_config(layout="centered")
file_prompt = st.text("Please select the file you want to run.")
file_selection = st.selectbox("File from week ", list(files.keys()))

if files[file_selection] != "":
    importlib.import_module(files[file_selection])

