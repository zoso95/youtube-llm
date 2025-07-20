# transcribe_worker.py
import sys
import whisper


def transcribe(audio_path, output_path):
    model = whisper.load_model("base", device="cpu")
    result = model.transcribe(audio_path)
    with open(output_path, "w") as f:
        f.write(result["text"])


if __name__ == "__main__":
    audio_path = sys.argv[1]
    output_path = sys.argv[2]
    transcribe(audio_path, output_path)
