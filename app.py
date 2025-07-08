import streamlit as st
import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# 불용어 설정
stop_words_en = set(stopwords.words('english'))
stop_words_ko = set(['그리고', '하지만', '그러나', '때문에', '위해', '또한', '및', '등'])  # 필요한 만큼 확장 가능

# Streamlit 설정
st.set_page_config(page_title="TF-IDF 키워드 추출기", layout="wide")
st.title("🧠 특허 텍스트 기반 TF-IDF 키워드 추출기 (spaCy 미사용 버전)")

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

        # 전처리 함수
        def preprocess(text):
            text = re.sub(r"[^가-힣a-zA-Z\s]", " ", text)  # 한글/영어만 남기기
            tokens = word_tokenize(text.lower())
            filtered = [w for w in tokens if len(w) > 1 and w not in stop_words_en and w not in stop_words_ko]
            return " ".join(filtered)

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

        # 다운로드
        csv = tfidf_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="⬇️ 키워드 결과 CSV 다운로드",
            data=csv,
            file_name='tfidf_keywords.csv',
            mime='text/csv'
        )
