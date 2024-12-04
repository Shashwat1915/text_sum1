from flask import Flask, render_template, request
from transformers import pipeline
from urllib.parse import quote as url_quote
import psutil
import os
import gc

pid = os.getpid()
python_process = psutil.Process(pid)

memory_use = python_process.memory_info().rss / 1024 ** 2
print(f"Memory usage: {memory_use:.2f} MB")


app = Flask(__name__)

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    input_text = request.form.get("text")

    if not input_text or len(input_text.split()) < 30:
        summary = "The input text is too short to summarize. Please provide a longer text."
    else:
        # Split text into chunks of up to 500 words
        chunks = [input_text[i:i+500] for i in range(0, len(input_text), 500)]
        summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
        summary = " ".join(summaries)
    
    return render_template("index.html", original_text=input_text, summary=summary)

    del summarizer
    gc.collect()
    
if __name__ == "__main__":
    app.run(debug=True, port=5001)
