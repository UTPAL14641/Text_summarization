import streamlit as st
from transformers import pipeline, set_seed
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
model_name = "https://github.com/UTPAL14641/Text_summarization/tree/main/pegasus-samsum-model"
tokenizer = AutoTokenizer.from_pretrained("https://github.com/UTPAL14641/Text_summarization/tree/main/tokenizer")
gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
def generate_summary(text):
    pipe = pipeline("summarization", model="pegasus-samsum-model",tokenizer=tokenizer)
    summary = pipe(text, **gen_kwargs)[0]["summary_text"]
    return summary

def main():
    st.title("Chat-text Summarizer")
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

