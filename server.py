from flask import Flask, request, render_template, jsonify
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

import nltk
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    if request.method == 'POST':
        files = {}
        for key in ['human_written', 'chatgpt_generated', 'gemini_generated', 'claude_generated']:
            uploaded_file = request.files[key]
            if uploaded_file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(file_path)
                files[key] = file_path

        if len(files) < 4:
            return 'Error: All four files are required.', 400

        # Get the selected similarity measure from the form
        similarity_measure = request.form.get('similarity_measure')

        # Extract and preprocess text from the uploaded PDF files
        human_written_doc = preprocess_text(extract_text_from_pdf(files['human_written']))
        ai_generated_docs = [
            preprocess_text(extract_text_from_pdf(files['chatgpt_generated'])),
            preprocess_text(extract_text_from_pdf(files['gemini_generated'])),
            preprocess_text(extract_text_from_pdf(files['claude_generated']))
        ]

        # Calculate similarity and dissimilarity based on user's choice
        similarity_results, tfidf_data = calculate_similarity_scores(human_written_doc, ai_generated_docs, similarity_measure)

        # Send similarity results and TF-IDF data as JSON
        return jsonify({
            'similarity_results': similarity_results,
            'tfidf_data': tfidf_data
        })

    return 'Invalid request method', 405

def extract_text_from_pdf(file_path):
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        text += f"\n[Error extracting text: {e}]"
    return text

lemmatizer = WordNetLemmatizer()

# Function to get hypernym (root meaning) of a word
def get_hypernym(word):
    synsets = wordnet.synsets(word)
    if not synsets:
        return word  # Return the word itself if no synsets are found
    
    # Get the first synset (most common meaning)
    synset = synsets[0]
    
    # Get the hypernym (more abstract/general word)
    hypernyms = synset.hypernyms()
    
    if hypernyms:
        # Return the first hypernym lemma (base form of the hypernym)
        return hypernyms[0].lemmas()[0].name()
    else:
        return word  # Return the word itself if no hypernyms are found

# Function to map POS tags to WordNet POS for lemmatization
def get_wordnet_pos(word):
    """
    Returns the WordNet POS tag that corresponds to the POS tag
    required by the WordNetLemmatizer.
    """
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove stop words
    stop_words = ENGLISH_STOP_WORDS
    words = [word for word in text.split() if word not in stop_words]
    
    # Lemmatize words using the correct POS tag and then get hypernyms
    processed_text = []
    for word in words:
        # Lemmatize the word
        lemmatized_word = lemmatizer.lemmatize(word, pos=get_wordnet_pos(word))
        
        # Get the hypernym (semantic root)
        root_word = get_hypernym(lemmatized_word)
        
        processed_text.append(root_word)
    
    # Join the words back into a string and remove extra whitespace
    return " ".join(processed_text)

def calculate_similarity_scores(human_written_doc, ai_generated_docs, similarity_measure):
    documents = [human_written_doc] + ai_generated_docs
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    similarity_results = []
    
    # Calculate similarity based on user's choice (Cosine or Jaccard)
    if similarity_measure == 'cosine':
        similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        dissimilarity_scores = 1 - similarity_scores
        for i, ai_name in enumerate(['ChatGPT', 'Gemini', 'claude AI']):
            similarity_results.append(f"Cosine Similarity between Human Written and {ai_name}: {similarity_scores[i]:.2f}")
            similarity_results.append(f"Cosine Dissimilarity between Human Written and {ai_name}: {dissimilarity_scores[i]:.2f}")
    
    elif similarity_measure == 'jaccard':
        def jaccard_similarity(v1, v2):
            intersection = np.sum(np.minimum(v1, v2) > 0)
            union = np.sum(np.maximum(v1, v2) > 0)
            return intersection / union if union != 0 else 0
        
        jaccard_sim = np.array([jaccard_similarity(tfidf_matrix[0].toarray().flatten(), tfidf_matrix[i].toarray().flatten()) for i in range(1, len(documents))])
        jaccard_dissim = 1 - jaccard_sim
        for i, ai_name in enumerate(['ChatGPT', 'Gemini', 'claude AI']):
            similarity_results.append(f"Jaccard Similarity between Human Written and {ai_name}: {jaccard_sim[i]:.2f}")
            similarity_results.append(f"Jaccard Dissimilarity between Human Written and {ai_name}: {jaccard_dissim[i]:.2f}")

    # Convert the TF-IDF matrix to a DataFrame for viewing the weights of each word
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=['Human Written', 'ChatGPT', 'Gemini', 'claude AI'], columns=vectorizer.get_feature_names_out())
    
    # Convert the DataFrame to dictionary to be sent as JSON
    tfidf_data = {
        'headers': list(tfidf_df.columns),
        'rows': tfidf_df.to_dict(orient='index')
    }

    return similarity_results, tfidf_data

if __name__ == '__main__':
    app.run(debug=True)
