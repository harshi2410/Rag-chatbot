# Conversational RAG Chatbot

A conversational chatbot built with Streamlit, regex-based intent matching, FAISS,
Sentence Transformers, and Hugging Face.

## Features
- Responds to greetings and help requests with pattern-matching rules
- Answers basic questions about popular book genres
- Upload a UTF-8 `.txt` file and ask questions about its contents
- Shows the document text used to answer each question
- Uses a clear fallback when a question is outside its supported topics or the document
  does not contain the answer
- Keeps the conversation visible during the Streamlit session

## Tech Stack
Python, Streamlit, Regex, FAISS, Sentence Transformers, Transformers

## Run the app

Install the dependencies from `requirements.txt`, then start Streamlit:

```bash
streamlit run app.py
python -m streamlit run app.py
```

The first document upload downloads the embedding model, and the first document
question also loads the answer-generation model. These models are cached while the
app is running.