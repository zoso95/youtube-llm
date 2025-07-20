A RAG LLM that summarizes a Youtube video and let's you run a Q&A based off the content. 

To install it. 

git clone https://github.com/zoso95/youtube-llm.git
cd youtube-llm
pip install -r requriments.txt
brew install ollama

# make sure ollama is running in the background
ollama run mistral

# local terminal version
python basic_implmentation.py 

# gradio server version

python app.py