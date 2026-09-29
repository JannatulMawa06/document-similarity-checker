# Document Similarity Checker (Web Version)

## What this is

This is a Flask web app I built that lets you upload two or more
documents and see how similar they are to each other, shown as a
percentage. It works by turning each document into a TF-IDF vector and
then computing the cosine similarity between every pair of documents.

It supports both `.txt` and `.pdf` files.

## Why I built it

I've been studying some basic NLP concepts in my coursework, mainly
tokenization, bag-of-words, and TF-IDF, and I wanted to actually build
something with them instead of just reading about the theory. I started
with a command-line script, turned it into a basic Flask app, and then
expanded it further with a nicer looking upload form and support for PDF
files, since most real documents I wanted to test (essays, assignments)
were PDFs rather than plain text files.

## How it works

1. You choose a file for "Document 1" and "Document 2" (and optionally
   add more documents with the "+ Add another document" link).
2. Each file is read in. `.txt` files are decoded directly, and `.pdf`
   files have their text pulled out page by page using the `pypdf`
   library.
3. The extracted text gets lowercased and cleaned up a little.
4. scikit-learn's `TfidfVectorizer` turns each document into a TF-IDF
   vector (this also handles tokenization and removes common English
   stop words like "the" and "and").
5. `cosine_similarity` measures how close each pair of vectors is.
6. The results page shows a similarity percentage and a bar for every
   pair of documents you uploaded.

## How to run it

Install the libraries this project needs:

```
pip install -r requirements.txt
```

Then start the app:

```
python app.py
```

Flask will print a local address, usually `http://127.0.0.1:5000`. Open
that in your browser, choose a file for Document 1 and Document 2 (add
more if you want), and click "Compare documents".

There are a few sample `.txt` files in the `sample_docs` folder you can
use to try it out right away. Feel free to also try it with your own
PDF files, like essays or lecture notes.

## Project structure

```
document_similarity_checker/
├── app.py
├── README.md
├── templates/
│   ├── index.html
│   └── results.html
└── sample_docs/
    ├── doc1_renewable_energy.txt
    ├── doc2_solar_power.txt
    └── doc3_pasta_recipe.txt
```

The sample docs are set up on purpose so that two of them (renewable
energy and solar power) are on related topics and should score higher
in similarity, while the third one (a pasta recipe) is unrelated and
should score much lower.

## Things I learned / found tricky

- Extracting text from a PDF is not as simple as reading a `.txt` file.
  I used the `pypdf` library, which lets you loop through each page of a
  PDF and pull out whatever text it can find. I learned that this
  doesn't always work perfectly, for example scanned PDFs (that are
  really just images of text) don't return any text at all, since there
  is no actual text layer to read.
- I also noticed that comparing a `.txt` file against a `.pdf` version of
  the exact same content doesn't give a perfect 100% match. This is
  because PDF text extraction can introduce small differences, like
  extra spaces or line breaks in different places, which slightly
  changes the TF-IDF vectors. It was a good reminder that "the same
  content" isn't always "the same text" once it's been through different
  file formats.
- Building the "add another document" button took some thought since I
  wanted every file input, including new ones added dynamically with
  JavaScript, to still work with the same Flask route. I ended up giving
  every file input the same `name="documents"` attribute, so Flask can
  collect all of them into one list with `request.files.getlist()`,
  no matter how many document slots the user adds.
