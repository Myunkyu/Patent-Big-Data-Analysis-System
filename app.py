import streamlit as st
import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# spaCy 모델 자동 설치
os.system("python -m spacy download en_core_web_sm")

# 웹 설정
st.set_page_config(page_title="TF-IDF 키워드 추출기", layout="wide")
st.title("🧠 특허 텍스트 기반 TF-IDF 키워드 추출기")

# 영어 자연어 처리 모델 불러오기
nlp = spacy.load("en_core_web_sm")

# 파일 업로드
uploaded_file = st.file_uploader("📤 엑셀(.xlsx) 파일을 업로드하세요", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ 파일 업로드 성공!")
    st.dataframe(df.head())

    # 열 선택
    selected_column = st.selectbox("🔎 TF-IDF 분석에 사용할 열을 선택하세요", df.columns)

    if selected_column:
        text_data = df[selected_column].astype(str).dropna().tolist()

        # 한글/영어 전처리 및 토큰 추출 함수
        def preprocess(text):
            # 언어 감지 없이 둘 다 처리 시도
            korean_tokens = re.findall(r"[가-힣]{2,}", text)
            english_tokens = [
                token.text for token in nlp(text)
                if token.pos_ in ['NOUN', 'VERB'] and len(token.text) > 1
            ]
            return " ".join(korean_tokens + english_tokens)

        # 전처리
        st.info("텍스트 전처리 및 TF-IDF 분석 중...")
        processed_texts = [preprocess(text) for text in text_data]

        # TF-IDF 분석
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(processed_texts)
        tfidf_scores = X.sum(axis=0).A1
        terms = vectorizer.get_feature_names_out()

        tfidf_df = pd.DataFrame({"Term": terms, "Score": tfidf_scores})
        tfidf_df = tfidf_df.sort_values(by="Score", ascending=False).head(30)

        # 결과 출력
        st.subheader("📋 상위 30개 키워드 (TF-IDF)")
        st.dataframe(tfidf_df)

        # 시각화
        st.subheader("📊 TF-IDF 키워드 시각화")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(tfidf_df["Term"][::-1], tfidf_df["Score"][::-1])
        ax.set_xlabel("TF-IDF Score")
        ax.set_ylabel("Keyword")
        st.pyplot(fig)

        # 다운로드 버튼
        csv = tfidf_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="⬇️ 키워드 결과 CSV 다운로드",
            data=csv,
            file_name='tfidf_keywords.csv',
            mime='text/csv'
        )
