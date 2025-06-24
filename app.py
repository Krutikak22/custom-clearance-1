import streamlit as st
from joblib import load
import pandas as pd
import time
import requests
import fitz  # PyMuPDF
import docx2txt
from PIL import Image
import pytesseract

st.set_page_config(page_title="Doc Classifier App", layout="wide")

st.title("📄 Intelligent Document Classifier")
st.markdown("<p style='color: gray;'>Upload or enter customs-related documents for classification and translation.</p>", unsafe_allow_html=True)

# === Load Model & Vectorizer ===
try:
    vectorizer = load('vectorizer.pkl')
    rf_model = load('rf_model.pkl')
except Exception as e:
    st.error(f"❌ Error loading model/vectorizer: {e}")
    st.stop()

# === LibreTranslate API
def libre_translate(text, target_lang):
    url = "https://libretranslate.de/translate"
    payload = {"q": text, "source": "auto", "target": target_lang, "format": "text"}
    response = requests.post(url, data=payload)
    if response.status_code == 200:
        return response.json()["translatedText"]
    return "⚠️ Translation failed."

# === Text Extraction Function
def extract_text_from_file(uploaded_file):
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        return " ".join(page.get_text() for page in doc)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return docx2txt.process(uploaded_file)
    elif file_type.startswith("image/"):
        img = Image.open(uploaded_file)
        return pytesseract.image_to_string(img)
    return ""

# === Tabs ===
tab1, tab2, tab3 = st.tabs(["🧠 Classification", "📝 Translate + Report", "ℹ️ About"])

# === Tab 1: Classification ===
with tab1:
    st.header("📂 Upload or Paste Document for Classification")

    col1, col2 = st.columns([1, 2])
    with col1:
        uploaded_file = st.file_uploader("Upload file (PDF, DOCX, Image)", type=["pdf", "docx", "jpg", "jpeg", "png"])
    with col2:
        user_input = st.text_area("Or paste text here:", height=200)

    if st.button("🔍 Classify"):
        text = ""
        if uploaded_file:
            with st.spinner("Extracting text from file..."):
                text = extract_text_from_file(uploaded_file)
        elif user_input.strip():
            text = user_input.strip()

        if not text:
            st.warning("Please upload a file or enter some text.")
        else:
            with st.spinner("Classifying..."):
                X_input = vectorizer.transform([text])
                rf_pred = rf_model.predict(X_input)[0]

            st.success(f"✅ Document classified as: **{rf_pred}**")
            st.text_area("📄 Extracted Text:", text, height=150)

# === Tab 2: Translation ===
with tab2:
    st.header("🌐 Translate Document Text")
    text_to_translate = st.text_area("Enter text to translate:", height=200)
    target_lang = st.selectbox("Select target language:", ["en", "es", "fr", "de", "hi", "ja", "zh", "ar"])

    if st.button("🌍 Translate & Download Report"):
        if not text_to_translate.strip():
            st.warning("Please enter some text.")
        else:
            with st.spinner("Translating and generating report..."):
                translated_text = libre_translate(text_to_translate, target_lang)
                report_df = pd.DataFrame({
                    "Original Text": [text_to_translate],
                    "Translated Text": [translated_text],
                    "Target Language": [target_lang]
                })
                report_csv = report_df.to_csv(index=False)

            st.success("✅ Translation Complete")
            st.text_area("Translated Text:", translated_text, height=200)
            st.download_button("📥 Download Report as CSV", data=report_csv, file_name="translation_report.csv", mime="text/csv")

# === Tab 3: About ===
with tab3:
    st.header("About this App")
    st.markdown("""
    This tool classifies documents relevant to **customs clearance** using a **Random Forest** model.

    Supported uploads:
    - 📄 PDFs
    - 📝 Word Documents (.docx)
    - 🖼️ Images (JPG/PNG)

    Bonus features:
    - 🌐 Translate extracted text using LibreTranslate
    - 📥 Download translation as a CSV report

    Built with ❤️ using **Streamlit**, **scikit-learn**, and **LibreTranslate**.
    """)
