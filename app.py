import os
import faiss

import gradio as gr
import pytubefix
import whisper

import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import requests
import subprocess

import random

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"  # Make sure it's downloaded


model = SentenceTransformer("all-MiniLM-L6-v2")


def get_transcript(video_url):
    print("Downloading video")
    data = pytubefix.YouTube(video_url)

    title = data.title

    os.makedirs("data", exist_ok=True)
    audio_file = f"{title}.m4a"
    audio_path = os.path.join("data", audio_file)
    transcript_path = os.path.join("data", f"{title}.txt")

    if os.path.exists(audio_path):
        print("Found existing downloaded audio, skipping")
    else:
        print(f"Downloading {title}")
        audio = data.streams.get_audio_only()
        audio.download(output_path="data/", filename=audio_file)

    if os.path.exists(transcript_path):
        print("Found cached transcript, skipping")
        with open(transcript_path, "r") as file:
            text = file.read()
        return title, text
    else:
        print("Transcribing")

        # Call the subprocess
        subprocess.run(
            ["python", "transcribe_worker.py", audio_path, transcript_path], check=True
        )

        # Read the result
        with open(transcript_path, "r") as f:
            text = f.read()

        return title, text


def make_vector_storage(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    chunks = [doc.page_content for doc in docs]

    # 2. Embed with sentence-transformers
    vectors = model.encode(chunks)

    # 3. Store in FAISS
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(np.array(vectors))

    return chunks, index


def retrieve(query, index, chunks, k=5):
    q_vector = model.encode([query])
    D, I = index.search(q_vector, k)
    return [chunks[i] for i in I[0]]


def make_prompt(context, question):
    context = "\n\n".join(context)
    prompt = f"""You are a helpful assistant, trying to answer Q&A from a transcript.

    Context:
    {context}

    Question:
    {question}

    First summarize the context, then answer the question.

    Answer:"""
    return prompt


def make_summary_prompt(text):
    prompt = f"""Could you please summarize this transcript?

    Transcript:
    {text}
    """
    return prompt


def summarize(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": make_summary_prompt(text), "stream": False},
    )

    return response.json()["response"]


def search_and_prompt(query, history, index, chunks):
    results = retrieve(query, index, chunks)
    prompt = make_prompt(results, query)
    response = requests.post(
        OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}
    )

    return response.json()["response"]


def transcribe_and_create_db(yt_video):
    title, text = get_transcript(yt_video)
    chunks, index = make_vector_storage(text)
    return f"Finished. Ready for Q&A for {title}", text, chunks, index


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🎧 Chat with a YouTube Video")

    text = gr.State()
    chunks = gr.State()
    index = gr.State()

    with gr.Row():
        yt_input = gr.Textbox(label="YouTube URL")
        status = gr.Textbox(label="Status")
        with gr.Column(scale=1):
            transcribe_btn = gr.Button("Transcribe")
            summarize_btn = gr.Button("Summarize")

    with gr.Row():
        summary = gr.Textbox(value="", label="Summary", interactive=False)

    transcribe_btn.click(
        fn=transcribe_and_create_db,
        inputs=yt_input,
        outputs=[status, text, chunks, index],
    )

    summarize_btn.click(fn=summarize, inputs=text, outputs=summary)

    gr.ChatInterface(
        fn=search_and_prompt, type="messages", additional_inputs=[index, chunks]
    )

demo.launch()
