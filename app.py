from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)
models = {
    "english": pipeline("summarization", model="facebook/bart-large-cnn"),
    "french": pipeline("summarization", model="facebook/mbart-large-50-many-to-many-mmt"),
    "spanish": pipeline("summarization", model="facebook/mbart-large-50-many-to-many-mmt"),
    "german": pipeline("summarization", model="facebook/mbart-large-50-many-to-many-mmt"),
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    input_text = request.form.get("text")
    language = request.form.get("language")
    
    if not input_text or len(input_text.split()) < 30:
        summary = "The input text is too short to summarize. Please provide a longer text."
    else:
        summarizer = models.get(language, models["english"]) 
        summary = summarizer(input_text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    
    return render_template("index.html", original_text=input_text, summary=summary, selected_language=language)

if __name__ == "__main__":
    app.run(debug=True, port=5001)