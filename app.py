import streamlit as st
from joblib import load
from googletrans import Translator  # pip install googletrans==4.0.0-rc1
import pandas as pd
import time

# Load models
vectorizer = load('tfidf_vectorizer.pkl')
rf_model = load('random_forest.pkl')
dt_model = load('decision_tree.pkl')
svm_model = load('svm_model.pkl')
translator = Translator()

st.set_page_config(page_title="Doc Classifier App", layout="wide")

st.title("📄 Intelligent Document Classifier")
st.markdown(
    "<p style='color: gray;'>Analyze, classify, and translate your documents easily.</p>",
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3 = st.tabs(["Classification 🧠", "Translation & Report 📝", "About ℹ️"])

with tab1:
    st.header("Classify your text")
    user_input = st.text_area(
        "Enter your document text:", height=200, placeholder="Paste your text here..."
    )

    if st.button("Classify"):
        if not user_input.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner('Analyzing...'):
                time.sleep(1)  # Simulate wait
                X_input = vectorizer.transform([user_input])
                rf_pred = rf_model.predict(X_input)[0]
                dt_pred = dt_model.predict(X_input)[0]
                svm_pred = svm_model.predict(X_input)[0]

            st.success("Classification Complete ✅")
            st.write(f"**🤖 Random Forest Prediction:** {rf_pred}")
            st.write(f"**🤖 Decision Tree Prediction:** {dt_pred}")
            st.write(f"**🤖 SVM Prediction:** {svm_pred}")

with tab2:
    st.header("Translation & Report")
    text_to_translate = st.text_area(
        "Enter text to translate:", height=200, placeholder="Type text to translate..."
    )
    target_lang = st.selectbox(
        "Select target language:",
        ["en", "es", "fr", "de", "hi", "ja", "zh-cn", "ar"],
    )

    if st.button("Translate & Download Report"):
        if not text_to_translate.strip():
            st.warning("Please enter some text!")
        else:
            with st.spinner('Translating and generating report...'):
                time.sleep(1)
                translated_text = translator.translate(
                    text_to_translate, dest=target_lang
                ).text

                report_df = pd.DataFrame(
                    {"Original Text": [text_to_translate], "Translated Text": [translated_text], "Target Language": target_lang}
                )
                report_csv = report_df.to_csv(index=False)

            st.success(f"✅ Translation Complete ({target_lang})")
            st.text_area("Translated Text:", translated_text, height=200)
            st.download_button(
                label="📥 Download Report as CSV",
                data=report_csv,
                file_name="translation_report.csv",
                mime="text/csv",
            )

with tab3:
    st.header("About this app")
    st.markdown(
        """
        This application lets you classify documents into categories using **machine learning models** 
        (Random Forest, Decision Tree, SVM) and also provides **text translation**.  
        
        **Features:**
        - Predict document type.
        - Translate content into multiple languages.
        - Download your reports as CSV files.
        - Easy and intuitive interface.

        Built with ❤️ using **Streamlit**.
        """
    )
