import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter
import os

# NLTK resource check
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

st.set_page_config(page_title="TF-IDF 키워드 추출기", layout="wide")
st.title("🧠 특허 텍스트 기반 TF-IDF 키워드 추출기 (언어별 불용어 적용)")

# 사용자 정의 불용어 리스트
patent_specific_korean_stopwords = set([
    '발명', '청구항', '구성', '기재', '도면', '장치', '포함', '특성', '방법', '단계',
    '발명자', '출원', '등록', '국제출원', '효력', '권리', '본원', '발명의', '위해',
    '된', '이용한', '하는', '할', '하는데', '에', '의', '로', '및', '다른', '일부', '여부',
    '발명의 명칭', '발명의 효과', '기술 분야', '배경 기술', '해결하려는 과제', '해결 수단',
    '이하', '상기한', '기술적', '요약', '하다', '되다', '상기', '를', '으로', '및', '에서', '과', '와', '하고'
])
patent_specific_english_stopwords = set([
    'invention', 'claim', 'embodiment', 'method', 'apparatus', 'device', 'system',
    'means', 'unit', 'step', 'portion', 'member', 'element', 'section', 'example',
    'fig', 'figure', 'said', 'wherein', 'thereof', 'therein', 'whereby', 'description',
    'preferred', 'technical', 'field', 'summary', 'background', 'disclosure',
    'advantages', 'features', 'object', 'objects', 'present', 'further', 'first', 'second', 'third'
])
default_en_stopwords = set(stopwords.words("english"))
default_ko_stopwords = set(['그리고', '하지만', '그러나', '때문에', '위해', '또한', '및', '등'])

uploaded_file = st.file_uploader("📤 엑셀(.xlsx) 파일을 업로드하세요", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.success("✅ 파일 업로드 성공!")
    st.dataframe(df.head())

    selected_column = st.selectbox("🔎 분석에 사용할 열을 선택하세요", df.columns)

    if selected_column:
        text_data = df[selected_column].astype(str).dropna().tolist()

        korean_texts, english_texts, all_tokens = [], [], []

        for text in text_data:
            try:
                lang = detect(text)
            except LangDetectException:
                lang = "unknown"

            if lang == "ko":
                tokens = re.findall(r"[가-힣]{2,}", text)
                filtered = [t for t in tokens if t not in patent_specific_korean_stopwords and t not in default_ko_stopwords]
                korean_texts.append(filtered)
                all_tokens.extend(filtered)

            elif lang == "en":
                tokens = re.sub(r"[^a-zA-Z\s]", " ", text).lower().split()
                filtered = [t for t in tokens if t not in patent_specific_english_stopwords and t not in default_en_stopwords and len(t) > 1]
                english_texts.append(filtered)
                all_tokens.extend(filtered)

        # 빈도 기반 불용어 추가
        word_freq = Counter(all_tokens)
        doc_count = len(korean_texts) + len(english_texts)
        dynamic_stopwords = {word for word, freq in word_freq.items() if freq / doc_count > 0.95}
        korean_texts = [[w for w in doc if w not in dynamic_stopwords] for doc in korean_texts]
        english_texts = [[w for w in doc if w not in dynamic_stopwords] for doc in english_texts]

        st.write(f"📌 한국어 문서 수: {len(korean_texts)}")
        st.write(f"📌 영어 문서 수: {len(english_texts)}")
        
        def perform_tfidf(texts):
            if not texts:
                return pd.DataFrame(columns=["Term", "Score"])
        
            # 공백 문서 제거
            joined = [" ".join(doc) for doc in texts if doc]
            joined = [doc for doc in joined if doc.strip() != ""]
        
            if not joined:
                return pd.DataFrame(columns=["Term", "Score"])
        
            tfidf = TfidfVectorizer(max_features=1000)
            X = tfidf.fit_transform(joined)
            scores = X.sum(axis=0).A1
            terms = tfidf.get_feature_names_out()
            df_result = pd.DataFrame({"Term": terms, "Score": scores})
            return df_result.sort_values(by="Score", ascending=False).head(30)

        st.subheader("📋 TF-IDF 키워드 추출 결과 (Korean)")
        df_ko = perform_tfidf(korean_texts)
        st.dataframe(df_ko)
        fig1, ax1 = plt.subplots()
        ax1.barh(df_ko["Term"][::-1], df_ko["Score"][::-1])
        fig1.tight_layout()
        st.pyplot(fig1)

        st.subheader("📋 TF-IDF 키워드 추출 결과 (English)")
        df_en = perform_tfidf(english_texts)
        st.dataframe(df_en)
        fig2, ax2 = plt.subplots()
        ax2.barh(df_en["Term"][::-1], df_en["Score"][::-1])
        fig2.tight_layout()
        st.pyplot(fig2)

        with pd.ExcelWriter("tfidf_results.xlsx") as writer:
            df_ko.to_excel(writer, sheet_name="Korean_TFIDF", index=False)
            df_en.to_excel(writer, sheet_name="English_TFIDF", index=False)

        with open("tfidf_results.xlsx", "rb") as f:
            st.download_button("⬇️ 결과 엑셀 다운로드", f.read(), file_name="tfidf_results.xlsx")
