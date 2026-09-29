"""
Document Similarity Checker (Web Version)
-------------------------------------------
This is a small Flask web app that lets you upload two or more documents
(.txt or .pdf files) and see how similar they are to each other. It uses
TF-IDF to turn each document into a vector, then uses cosine similarity
to compare them.

I started this as a command-line tool, turned it into a basic Flask app,
and then expanded it further to support PDF files and a nicer looking
upload form with a separate slot for each document.
"""

from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
import io

app = Flask(__name__)


def clean_text(text):
    text = text.lower()
    text = text.strip()
    return text


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + " "
    return full_text


def get_text_from_file(file):
    filename = file.filename.lower()

    if filename.endswith(".txt"):
        raw_bytes = file.read()
        return raw_bytes.decode("utf-8")

    elif filename.endswith(".pdf"):
       
        file_bytes = io.BytesIO(file.read())
        return extract_text_from_pdf(file_bytes)

    else:
       
        return None


@app.route("/", methods=["GET"])
def home():
    
    return render_template("index.html")


@app.route("/compare", methods=["POST"])
def compare():
    uploaded_files = request.files.getlist("documents")

    
    actual_files = []
    for file in uploaded_files:
        if file.filename != "":
            actual_files.append(file)

    if len(actual_files) < 2:
        error_message = "Please choose at least two documents to compare."
        return render_template("index.html", error=error_message)

    documents = []
    file_names = []

    
    for file in actual_files:
        filename = file.filename

        if not (filename.lower().endswith(".txt") or filename.lower().endswith(".pdf")):
            error_message = filename + " is not a .txt or .pdf file."
            return render_template("index.html", error=error_message)

        try:
            raw_text = get_text_from_file(file)
        except Exception as error:
            error_message = "Could not read file " + filename + ": " + str(error)
            return render_template("index.html", error=error_message)

        if raw_text is None or raw_text.strip() == "":
            error_message = "Could not find any readable text in " + filename + "."
            return render_template("index.html", error=error_message)

        cleaned = clean_text(raw_text)
        documents.append(cleaned)
        file_names.append(filename)

    
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(documents)

   
    similarity_matrix = cosine_similarity(tfidf_matrix)

   
    results = []
    num_docs = len(file_names)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            score = similarity_matrix[i][j]
            percentage = round(score * 100, 2)
            results.append({
                "doc_a": file_names[i],
                "doc_b": file_names[j],
                "percentage": percentage,
                "badge": None
            })

    
    if len(results) > 1:
        best_match = max(results, key=lambda r: r["percentage"])
        worst_match = min(results, key=lambda r: r["percentage"])
        best_match["badge"] = "Highest match"
        if worst_match is not best_match:
            worst_match["badge"] = "Most different"

    return render_template("results.html", results=results, doc_count=num_docs)


if __name__ == "__main__":
    app.run(debug=True)
