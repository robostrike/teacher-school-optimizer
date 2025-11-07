import streamlit as st

st.title("Teacher Allocation Optimizer")

st.write("👋 Welcome! Upload your teacher and school data below to get started.")

teachers = st.file_uploader("Upload Teachers CSV", type="csv")
schools = st.file_uploader("Upload Schools CSV", type="csv")
stations = st.file_uploader("Upload Stations Graph CSV", type="csv")

if teachers and schools and stations:
    st.success("✅ All files uploaded. Optimization tools will appear here.")
else:
    st.info("Please upload all three files to continue.")