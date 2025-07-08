# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="특허 데이터 업로드", layout="wide")

st.title("📄 특허 데이터 업로드 시스템")

# 파일 업로드
uploaded_file = st.file_uploader("엑셀 파일을 업로드하세요 (.xlsx)", type=["xlsx"])

# 업로드된 경우
if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file, engine="openpyxl")
        st.success("✅ 파일 업로드 성공!")
        st.dataframe(df)  # 업로드한 데이터 미리보기
    except Exception as e:
        st.error(f"❌ 오류 발생: {e}")
else:
    st.info("⏳ 왼쪽에서 엑셀 파일을 선택하면 미리보기가 여기에 표시됩니다.")
