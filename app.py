import streamlit as st
from transformers import pipeline

# Load summarization and title generation pipelines
@st.cache_resource
def load_pipelines():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    title_generator = pipeline("text2text-generation", model="google/flan-t5-small")  # small but effective
    return summarizer, title_generator

summarizer, title_generator = load_pipelines()

# Streamlit UI
st.title("📝 Blog Title & Summary Generator")

input_text = st.text_area("Paste your blog paragraph below:", height=250)

if st.button("Generate Title & Summary"):
    if input_text.strip():
        # Generate Summary
        summary_output = summarizer(input_text, max_length=80, min_length=25, do_sample=False)
        summary = summary_output[0]['summary_text']

        # Generate Title (we ask for a title based on summary)
        prompt = f"Generate a blog title for: {summary}"
        title_output = title_generator(prompt, max_length=15)[0]['generated_text']

        st.subheader("📌 Title:")
        st.write(title_output.strip())

        st.subheader("📝 Summary:")
        st.write(summary.strip())
    else:
        st.warning("Please paste a blog paragraph to generate title and summary.")
