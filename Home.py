import streamlit as st

st.set_page_config(page_title="Generate Synthetic Data", layout="wide")

st.title("Generate Synthetic Data")
st.write("Use the left sidebar to open a model page.")

st.markdown("""
### Available pages
- CTGAN Generator
- CTGAN + Decision Tree
- CTGAN + SVR
- CTGAN + XGBoost     
- WGAN Generator
- WGAN + Decision Tree
- WGAN + SVR
- WGAN + XGBoost
- CTGAN + XGBoost
""")