import streamlit as st
import pandas as pd

"""
streamlit_app.py - Base opening Streamlit app

Place this file at the project root and run:
    streamlit run streamlit_app.py
"""


APP_TITLE = "Teacher School Optimizer"
APP_ICON = "🏫"

def configure_page():
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        layout="wide",
        initial_sidebar_state="expanded",
    )

def render_sidebar():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Upload Data", "Settings"])
    st.sidebar.markdown("---")
    st.sidebar.caption("Dev container: Ubuntu 24.04.2 LTS")
    return page

def render_home():
    st.title(APP_TITLE)
    st.header("Welcome")
    st.write(
        "This is a starter Streamlit app for the Teacher School Optimizer project. "
        "Use the sidebar to navigate. Implement features for data upload, "
        "optimization and visualization in their respective pages."
    )

def render_upload():
    st.header("Upload Data")
    st.write("Upload CSV files to begin.")
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            st.dataframe(df.head())
            st.session_state["last_df"] = df
        except Exception as e:
            st.error(f"Error reading file: {e}")

def render_settings():
    st.header("Settings")
    st.write("Configure app-level options.")
    st.checkbox("Enable verbose logging", key="verbose")
    st.number_input("Max results", min_value=1, max_value=1000, value=100, step=1, key="max_results")

def main():
    configure_page()
    page = render_sidebar()

    if page == "Home":
        render_home()
    elif page == "Upload Data":
        render_upload()
    elif page == "Settings":
        render_settings()
    else:
        st.info("Select a page from the sidebar.")

if __name__ == "__main__":
    main()