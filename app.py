import streamlit as st
import importlib

files = {
    "---Please select---": "", 
    "Week 2": "Week2",
    "Week 3": "Week3"
}
file_list = list(files.keys())

st.set_page_config(page_title="Jesse Xu's Senior Project", page_icon="favicon.ico", layout="centered")
query_params = st.experimental_get_query_params()
st.header("Jesse Xu's Senior Project")
st.subheader("Make Computers See: Image Processing with OpenCV")
st.markdown("***")
file_prompt = st.markdown("Please select a week number to see my work of the week")

file_index = 0
if "file_selection" in query_params:
    file_index = file_list.index(query_params["file_selection"][0])

file_selection = st.selectbox("Week number", file_list, index=file_index)
st.markdown("***")

if files[file_selection] != "":
    query_params["file_selection"] = file_selection
    st.experimental_set_query_params(**query_params)
    module = __import__(files[file_selection])
    getattr(module, "main")()
