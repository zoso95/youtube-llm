import pytubefix
import whisper
import code
import os
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import requests

VIDEO = "https://www.youtube.com/watch?v=geIhl_VE0IA"
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
        return text
    else:
        print("Transcribing")
        model = whisper.load_model("base")
        text = model.transcribe(audio_path)
        with open(transcript_path, "w") as file:
            file.write(text["text"])

        return text["text"]


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


def retrieve(query, index, chunks, k=5):
    q_vector = model.encode([query])
    D, I = index.search(q_vector, k)
    return [chunks[i] for i in I[0]]


def search_and_prompt(query, index, chunks):
    results = retrieve(query, index, chunks)
    prompt = make_prompt(results, query)
    response = requests.post(
        OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False}
    )

    return response.json()["response"]


def main():

    text = get_transcript(VIDEO)
    chunks, index = make_vector_storage(text)

    response = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": make_summary_prompt(text), "stream": False},
    )

    print(response.json()["response"])

    print("RAG Server is running. Type your query below (or 'exit'):\n")

    while True:
        q = input(">> ")
        if q.lower() in {"exit", "quit"}:
            break
        try:
            answer = search_and_prompt(q, index, chunks)
            print(f"\n🧠 {answer}\n")
        except Exception as e:
            print("❌ Error:", e)


if __name__ == "__main__":
    main()
