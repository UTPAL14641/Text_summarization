import streamlit as st
from transformers import pipeline, set_seed
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = r"C:\Users\Win10\Desktop\text_summ_streamlit\pegasus-samsum-model"
tokenizer = AutoTokenizer.from_pretrained(r"C:\Users\Win10\Desktop\text_summ_streamlit\tokenizer")
gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
def generate_summary(text):
    pipe = pipeline("summarization", model="pegasus-samsum-model",tokenizer=tokenizer)
    summary = pipe(text, **gen_kwargs)[0]["summary_text"]
    return summary

def main():
    st.title("Text Summarization with PEGASUS-CNN_Dailymail")
    st.write("Enter the text to be summarized:")
    text = st.text_area("Input Text", height=200)
    if st.button("Summarize"):
        if text.strip() != '':
            summary = generate_summary(text)
            st.subheader("Summary:")
            st.write(summary)
        else:
            st.warning("Please enter some text.")

if __name__ == '__main__':
    main()
