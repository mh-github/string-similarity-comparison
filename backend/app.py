from flask import Flask, render_template, request, jsonify
from similarity_methods import *
import gensim.downloader as api
import spacy
import gensim
from sentence_transformers import SentenceTransformer

app            = Flask(__name__)
word_vectors   = api.load("glove-wiki-gigaword-100")  # load pre-trained word-vectors from gensim-data
nlp            = spacy.load("en_core_web_md")  # make sure to use larger package!
# Load pre-trained Word2Vec model
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Load hugging face model
hugging_face_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/submit", methods=["POST"])
def submit():
    data = request.get_json()
    input_text1 = data.get("inputText1", "")
    input_text2 = data.get("inputText2", "")

    # Process the input as needed, for now, just echo it back
    # return jsonify({"message": f"Received: {input_text1} and {input_text2}"})
    string1 = input_text1
    string2 = input_text2
    cosine_similarity       = calc_cosine_similarity(input_text1, input_text2)
    gensim_similarity       = calc_gensim_similarity(input_text1, input_text2, word_vectors)
    spacy_similarity        = calc_spacy_similarity(input_text1, input_text2, nlp)
    google_similarity       = calc_google_similarity(input_text1, input_text2, word2vec_model)
    hugging_face_similarity = calc_hugging_face_similarity(input_text1, input_text2, hugging_face_model)

    print(f"cosine_similarity = {cosine_similarity}")
    print(f"cosine_similarity type = {type(cosine_similarity)   }")
    print(f"gensim_similarity = {gensim_similarity}")
    print(f"gensim_similarity type = {type(gensim_similarity)}")
    print(f"spacy_similarity = {spacy_similarity}")
    print(f"spacy_similarity type = {type(spacy_similarity)}")
    print(f"google_similarity = {google_similarity}")
    print(f"hugging_face_similarity = {hugging_face_similarity}")

    return jsonify({
        "string1": string1,
        "string2": string2,
        "cosine_similarity": cosine_similarity,
        "gensim_similarity": gensim_similarity,
        "spacy_similarity": spacy_similarity,
        "google_similarity": google_similarity,
        "hugging_face_similarity": hugging_face_similarity
    })

if __name__ == "__main__":
    app.run(debug=True)
