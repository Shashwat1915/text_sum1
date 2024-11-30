from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Initialize the text summarization pipeline with DistilBART model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/summarize", methods=["POST"])
def summarize():
    input_text = request.form.get("text")

    if not input_text or len(input_text.split()) < 30:
        summary = "The input text is too short to summarize. Please provide a longer text."
    else:
        # Generate the summary
        summary = summarizer(input_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
    
    return render_template("index.html", original_text=input_text, summary=summary)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
