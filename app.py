import chromadb
import torch
import html
import json
from flask import Flask, render_template, request
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from typing import Dict

app = Flask(__name__)

# Set hugging face access token
with open("token.txt", "r") as f:
    token = f.readline().strip()

# Set hugging face access token
with open("database_path.txt", "r") as f:
    db_path = f.readline().strip()

# Set torch device and connect to chromaDB
device = torch.device("mps" if torch.mps.is_available() else "cpu")
client = chromadb.PersistentClient(path=db_path)
collection = client.get_collection(name="arxiv")

# Load Tuned Model from Hugging Face: Fine-tuned BERT for classification
tokenizer = AutoTokenizer.from_pretrained("MODEL REPO NAME", token=token)
model = AutoModelForSequenceClassification.from_pretrained(
    "MODEL REPO NAME", token=token
).to(device)

# Load the label2id mapping from the JSON file
with open("label2id_1000.json", "r") as f:
    label2id = json.load(f)
id2label = {v: k for k, v in label2id.items()}
model.config.id2label = id2label


def get_journal_name(title, abstract):
    combined_text = f"{title} {abstract}"  # Combine title and abstract
    inputs = tokenizer(
        combined_text, return_tensors="pt", truncation=True, padding=True
    ).to(device)
    outputs = model(**inputs)
    label_id = torch.argmax(outputs.logits, dim=1).item()
    label = model.config.id2label[label_id]
    return f"Our Journal Recommendation: {html.unescape(label)}"


def query_db(title: str, abstract: str) -> list[Dict]:
    # Returns a list of dicts with keys: doi, journal, title
    query_result = collection.query(query_texts=[f"{title} {abstract}"], n_results=10)
    result = []
    # Get rid of symbols like &amp so it renders better.
    for item in query_result["metadatas"][0]:
        item["journal"] = html.unescape(item["journal"])
        result.append(item)
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    title = ""
    abstract = ""
    results = []
    label = ""
    error = None

    if request.method == "POST":
        title = request.form.get("title")
        abstract = request.form.get("abstract")

        if not title or not abstract:
            error = "Both title and abstract are required."
        else:
            results = query_db(title, abstract)
            label = get_journal_name(title, abstract)

    return render_template(
        "index.html",
        title=title,
        abstract=abstract,
        results=results,
        label=label,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
