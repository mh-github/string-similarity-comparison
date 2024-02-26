import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import numpy as np

def calc_cosine_similarity(string1, string2):
    print(f"Processing cosine similarity for {string1} and {string2}")
    # Tokenization
    string1_tokens = re.findall(r'\w+', string1.lower())
    # print(string2)
    string2_tokens = re.findall(r'\w+', string2.lower())

    # print(jd_tokens)
    # print(profile_tokens)

    if len(string1_tokens[0]) == 1 or len(string2_tokens[0]) == 1:
        if string1_tokens[0] in string2_tokens or string2_tokens[0] in string1_tokens:
            return True
        else:
            return False

    # Stemming
    stemmer = PorterStemmer()
    string1_tokens = [stemmer.stem(word) for word in string1_tokens]
    string2_tokens = [stemmer.stem(word) for word in string2_tokens]

    # print(jd_tokens)
    # print(profile_tokens, '###')

    # Feature extraction (TF-IDF)
    vectorizer = TfidfVectorizer()
    string1_vector = vectorizer.fit_transform([" ".join(string1_tokens)])
    string2_vector = vectorizer.transform([" ".join(string2_tokens)])

    similarity = cosine_similarity(string1_vector, string2_vector)
    return float(similarity[0][0])

def calc_gensim_similarity(string1, string2, word_vectors):
    print(f"Processing gensim similarity for {string1} and {string2}")

    try:
        similarity = word_vectors.similarity(string1, string2)
    except Exception as e:
        return str(e)
    
    # Before returning similarity[0][0], check if similarity is scalar
    if np.isscalar(similarity):
        return float(similarity)
    else:
        return float(similarity[0][0])

def calc_spacy_similarity(string1, string2, nlp):
    print(f"Processing spacy similarity for {string1} and {string2}")

    doc1 = nlp(string1)
    doc2 = nlp(string2)

    similarity = doc1.similarity(doc2)
    return similarity

# Function to compute average vector for a document
def document_vector(word2vec_model, doc):
    doc = [word for word in doc if word in word2vec_model.key_to_index]
    
    # Check if the filtered list is empty
    if not doc:
        # Return a zero vector if no words in the document are in the model's vocabulary
        return np.zeros(word2vec_model.vector_size)
    
    # Otherwise, return the mean of the vectors for the words in the document
    return np.mean([word2vec_model.get_vector(word) for word in doc], axis=0)
   
def calc_google_similarity(string1, string2, word2vec_model):
    print(f"Processing google similarity for {string1} and {string2}")

    # Tokenize and preprocess the documents
    string1_tokens = string1.lower().split()
    string2_tokens = string2.lower().split()

    # Compute document vectors
    string1_vector = document_vector(word2vec_model, string1_tokens)
    string2_vector = document_vector(word2vec_model, string2_tokens)

    # Ensure the vectors are 2D arrays with shape (1, N)
    string1_vector_2d = string1_vector.reshape(1, -1)
    string2_vector_2d = string2_vector.reshape(1, -1)

    # Compute cosine similarity between document vectors
    similarity = cosine_similarity(string1_vector_2d, string2_vector_2d)[0][0]

    return float(similarity)
                 
def calc_hugging_face_similarity(string1, string2, hugging_face_model):
    print(f"Processing hugging face similarity for {string1} and {string2}")
    sentences  = [string1, string2]
    embeddings = hugging_face_model.encode(sentences)

    # Save embeddings to a JSON file, not needed but to Analyse data we saved to file
    with open('embeddings.json', 'w') as f:
        json.dump(embeddings.tolist(), f)

    # Convert lists of vectors to numpy arrays
    vectors1 = np.array(embeddings.tolist()[0])
    vectors2 = np.array(embeddings.tolist()[1])

    # Reshape the arrays if needed
    if len(vectors1.shape) == 1:
        vectors1 = vectors1.reshape(1, -1)
    if len(vectors2.shape) == 1:
        vectors2 = vectors2.reshape(1, -1)

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(vectors1, vectors2)
    return float(similarity_matrix[0][0])