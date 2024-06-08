import io
import streamlit as st
from pdfminer.high_level import extract_text
from transformers import T5ForConditionalGeneration, T5Tokenizer

upload = st.file_uploader(
    label = "File Upload",
    type = ["pdf"],
    accept_multiple_files = False,
    help = "Upload file to generate questions based on."
)

# convert pdf into paragraphs
def get_paragraphs(pdf):
    extracted_text = extract_text(pdf)

    paragraphs = extracted_text.split('\n\n')
    result_paragraphs = []
    current_paragraph = ""
    for paragraph in paragraphs:
        # split paragraph into words
        words = paragraph.split()
        for word in words:
            current_paragraph += word + " "
            # check if the current paragraph exceeds the word limit
            if len(current_paragraph.split()) >= 1000:
                result_paragraphs.append(current_paragraph.strip())
                current_paragraph = ""
    # append any remaining text as a paragraph
    if current_paragraph:
        result_paragraphs.append(current_paragraph.strip())
    return result_paragraphs
    

if (upload is not None):
    paragraphs = get_paragraphs(upload)

    # load model and tokenizer
    dir = "fine-tuned"
    model = T5ForConditionalGeneration.from_pretrained(dir)
    tokenizer = T5Tokenizer.from_pretrained(dir)

    questions = []

    # generate questions paragraph by paragraph
    for paragraph in paragraphs:
        context = paragraph
        input_tokens = tokenizer.encode(context, return_tensors = "pt")

        output_tokens = model.generate(
            input_tokens,
            num_return_sequences = 3,
            do_sample = True,
            top_k = 25,  
            max_length = 50,  
            temperature = 0.7,
        )

        for output_token in output_tokens:
            output = tokenizer.decode(output_token, skip_special_tokens=True)
            questions.append(output)

    with st.chat_message("assistant"):
        st.write('**Generated questions:**')
        for question in questions:
            st.write(f'- {question}')