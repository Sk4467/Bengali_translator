import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer from Hugging Face Model Hub
MODEL_NAME = "Sk4467/Bengali_translator"  # Replace with your model's path on Hugging Face
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

st.title("English to Bengali Translation")

# User input
english_text = st.text_area("Enter English text:", "")
st.button("Submit")

if english_text:
    # Encode the text and generate translation
    encoded = tokenizer.encode(english_text, return_tensors="pt")
    translation_ids = model.generate(encoded)
    bengali_translation = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
    
    # Display the translation
    st.write("Bengali Translation:")
    st.write(bengali_translation)
st.caption("Made with ❤ by Chad ")
