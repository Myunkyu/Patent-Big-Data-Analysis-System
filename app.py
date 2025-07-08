import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from collections import Counter

nltk.download("punkt")
nltk.download("stopwords")

st.set_page_config(page_title="TF-IDF 키워드 추출기", layout="wide")
st.title("🧠 특허 텍스트 기반 TF-IDF 키워드 추출기 (언어별 불용어 적용)")

# 사용자 정의 불용어 리스트 (예시)
patent_specific_korean_stopwords = set([
    '발명', '청구항', '구성', '기재', '도면', '장치', '포함', '특성', '방법', '단계', '출원', '등록', '효력',
    '권리', '기술', '분야', '해결', '수단', '된', '이용한', '하는', '할', '에', '의', '로', '및', '에서', '과', '와'
])
patent_specific_english_stopwords = set([
    'invention', 'claim', 'embodiment', 'method', 'apparatus', 'device', 'system',
    'means', 'unit', 'step', 'portion', 'member', 'element', 'section', 'example',
    'fig', 'figure', 'said', 'wherein', 'thereof', 'therein', 'whereby', 'description',
    'preferred', 'technical', 'field', 'summary', 'background', 'disclosure',
    'advantages', 'features', 'object', 'objects', 'present', 'further'
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
            except:
                lang = "unknown"

            if lang == "ko":
                tokens = re.findall(r"[가-힣]{2,}", text)
                filtered = [t for t in tokens if t not in patent_specific_korean_stopwords and t not in default_ko_stopwords]
                korean_texts.append(filtered)
                all_tokens.extend(filtered)

            elif lang == "en":
                tokens = word_tokenize(re.sub(r"[^a-zA-Z\s]", " ", text).lower())
                filtered = [t for t in tokens if t not in patent_specific_english_stopwords and t not in default_en_stopwords and len(t) > 1]
                english_texts.append(filtered)
                all_tokens.extend(filtered)

        # 빈도 기반 불용어 추가
        word_freq = Counter(all_tokens)
        doc_count = len(korean_texts) + len(english_texts)
        dynamic_stopwords = {word for word, freq in word_freq.items() if freq / doc_count > 0.8}
        korean_texts = [[w for w in doc if w not in dynamic_stopwords] for doc in korean_texts]
        english_texts = [[w for w in doc if w not in dynamic_stopwords] for doc in english_texts]

        def perform_tfidf(texts):
            if not texts:
                return pd.DataFrame(columns=["Term", "Score"])
            joined = [" ".join(doc) for doc in texts]
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
        st.pyplot(fig1)

        st.subheader("📋 TF-IDF 키워드 추출 결과 (English)")
        df_en = perform_tfidf(english_texts)
        st.dataframe(df_en)
        fig2, ax2 = plt.subplots()
        ax2.barh(df_en["Term"][::-1], df_en["Score"][::-1])
        st.pyplot(fig2)

        with pd.ExcelWriter("tfidf_results.xlsx") as writer:
            df_ko.to_excel(writer, sheet_name="Korean_TFIDF", index=False)
            df_en.to_excel(writer, sheet_name="English_TFIDF", index=False)

        with open("tfidf_results.xlsx", "rb") as f:
            st.download_button("⬇️ 결과 엑셀 다운로드", f.read(), file_name="tfidf_results.xlsx")
