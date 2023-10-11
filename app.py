

import pandas as pd
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np
import re
import ast
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from flask import Flask, jsonify

import pymysql

host = 'localhost'
user = 'root'
password = ''
database = 'cvd'

connection = pymysql.connect(
    host=host, user=user, password=password, database=database)
query = 'SELECT * FROM videos'
querycredits = 'SELECT * FROM credits'
moviess = pd.read_csv('./videos.csv')
creditss = pd.read_csv('./credits.csv')

movies = pd.read_sql(query, connection)
movies.head(2)

movies.shape

credits = pd.read_sql(querycredits, connection)
credits.head()

movies = movies.merge(credits, on='title')
movies['movie_id'] = movies['id']
movies.head()


movies = movies[['movie_id', 'title', 'overview',
                 'subject', 'keywords', 'cast', 'crew']]
movies.head()


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


movies.dropna(inplace=True)

movies['subject'] = movies['subject'].apply(convert)
movies.head()

movies['keywords'] = movies['keywords'].apply(convert)

movies.head()

ast.literal_eval(
    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L


movies['cast'] = movies['cast'].apply(convert)
movies.head()

movies['cast'] = movies['cast'].apply(lambda x: x[0:3])


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


movies['crew'] = movies['crew'].apply(fetch_director)

# movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1


movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['subject'] = movies['subject'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

movies.head()

movies['overview'] = movies['overview'].apply(lambda x: x.split())

mv = movies
print(mv)

movies['tags'] = movies['overview'] + movies['subject'] + \
    movies['keywords'] + movies['cast'] + movies['crew']


new = movies
# new.head()

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()


class CountVectorizerNormal:
    def __init__(self, lowercase=True, token_pattern=r"(?u)\b\w\w+\b", weight=None):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.vocabulary = defaultdict(int)
        self.stop_words = set()

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def fit(self, raw_documents):
        for doc in raw_documents:
            tokens = self._tokenize(doc)
            for token in tokens:
                self.vocabulary[token] += 1

    def transform(self, raw_documents):
        rows, cols, data = [], [], []
        for i, doc in enumerate(raw_documents):
            print(doc)
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocabulary and token not in self.stop_words:
                    rows.append(i)
                    cols.append(self.vocabulary[token])
                    data.append(1)
        X = csr_matrix((data, (rows, cols)), shape=(
            len(raw_documents), len(self.vocabulary)))
        return X

    def _tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(self.token_pattern, text)
        return tokens


class CountVectorizer:
    def __init__(self, lowercase=True, token_pattern=r"(?u)\b\w\w+\b", column_weights=None, spx=0):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.vocabulary = defaultdict(int)
        self.stop_words = set()
        self.column_weights = column_weights if column_weights is not None else {}
        self.spx = spx

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def fit(self, raw_documents):
        for doc in raw_documents:
            tokens = self._tokenize(doc)
            for token in tokens:
                self.vocabulary[token] += 1

    def transform(self, raw_documents):
        rows, cols, data = [], [], []
        for i, doc in enumerate(raw_documents):
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocabulary and token not in self.stop_words:
                    weight = self.column_weights.get(token, self.spx)
                    rows.append(i)
                    cols.append(self.vocabulary[token])
                    data.append(weight)
        X = csr_matrix((data, (rows, cols)), shape=(
            len(raw_documents), len(self.vocabulary)))
        return X

    def _tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(self.token_pattern, text)
        return tokens

import re
from collections import defaultdict
from scipy.sparse import csr_matrix
import pandas as pd

class CountVectorizerAllColumns:
    def __init__(self, lowercase=True, token_pattern=r"(?u)\b\w\w+\b", column_weights=None):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.vocabulary = defaultdict(int)
        self.stop_words = set()
        self.column_weights = column_weights if column_weights is not None else {}

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def fit(self, df):
        self.column_names = df.columns
        for column_name in df.columns:
            for cell in df[column_name]:
                if isinstance(cell, str):
                    tokens = self._tokenize(cell)
                    for token in tokens:
                        self.vocabulary[token] += 1

    def transform(self, df):
        rows, cols, data = [], [], []
        current_row = 0  # Keep track of the current row index
        for i, row in df.iterrows():
            for column_name in self.column_names:
                cell = row[column_name]
                if isinstance(cell, str):
                    tokens = self._tokenize(cell)
                    for token in tokens:
                        weight = self.column_weights.get(column_name, 1.0)
                        if token in self.vocabulary:
                            rows.append(current_row)  # Use current_row as the row index
                            cols.append(self.vocabulary[token])
                            data.append(weight)
            current_row += 1  # Increment the row index for the next document
    
        num_rows = len(df)
        num_columns = len(self.vocabulary)
        X = csr_matrix((data, (rows, cols)), shape=(num_rows, num_columns))
        return X

    def _tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(self.token_pattern, text)
        return tokens








df = pd.DataFrame(movies)

# Define column weights (adjust as needed)
column_weights = {'cast': 1.1, 'crew': 1.4, 'tags': 1.2,
                  'keywords': 1.2, 'subject': 3, "overview": 1}
column_weights_main = {'cast': 1.19, 'crew': 1.275, 'tags': 1.2, 'keywords': 1.1, 'subject': 1, "overview":1}

# # Initialize and use CountVectorizerJaccardMultiColumn
# vectorizer = CountVectorizerJaccardMultiColumn()
# X = vectorizer.fit_transform(df)


cv = CountVectorizer(column_weights=column_weights, spx=.9)
cvn = CountVectorizerAllColumns(column_weights=column_weights_main)




# Fit the vectorizer on the 'tags' column of the 'new' DataFrame

tfidf_matrixn = cvn.fit_transform(movies)



print(tfidf_matrix)
# Compute Pearson similarity matrix

similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
similarity_matrixn = cosine_similarity(tfidf_matrixn, tfidf_matrixn)


num_docs, num_terms = tfidf_matrix.shape
document_sets = []

for i in range(num_docs):
    # Get the non-zero indices (terms with non-zero TF-IDF values)
    non_zero_indices = np.nonzero(tfidf_matrix[i])[1]
    # Create a set of terms for the document
    document_set = set(non_zero_indices)
    document_sets.append(document_set)

 # Since Jaccard similarity is symmetric
# def jaccard_similarity(set1, set2):
#     intersection = len(set1 & set2)
#     union = len(set1 | set2)
#     return intersection / union

# # Calculate Jaccard similarities between documents
# num_docs = len(document_sets)
# jaccard_similarities = np.zeros((num_docs, num_docs))

# for i in range(num_docs):
#     for j in range(i, num_docs):
#         jaccard_similarities[i, j] = jaccard_similarity(document_sets[i], document_sets[j])
#         jaccard_similarities[j, i] = jaccard_similarities[i, j]
# # Print the Jaccard similarities


# Calculate the cosine similarity matrix
similarity_matrix_p = cosine_similarity(tfidf_matrix, tfidf_matrix)
similarity_matrix_n = cosine_similarity(tfidf_matrixn, tfidf_matrixn)


# Subtract the mean from each document's TF-IDF values
# centered_tfidf = tfidf_matrix - mean_tfidf

# # Calculate the Pearson Correlation Coefficient (PCC) similarity matrix

# print("Shape of tfidf_matrix:", tfidf_matrix.shape)
# print("Shape of mean_tfidf:", mean_tfidf.shape)
# print("Data type of tfidf_matrix:", tfidf_matrix.dtype)
# print("Data type of mean_tfidf:", mean_tfidf.dtype)
# Subtract the mean from each document's TF-IDF values


# Calculate the Pearson Correlation Coefficient (PCC) similarity matrix
# pcc_similarity_matrix = np.corrcoef(centered_tfidf)
# print(similarity_matrix)
# print(pcc_similarity_matrix)
def generate_recommendations(similarity_matrix, movie_id, top_k):
    # Find the index of the movie with the given title
    movie_index = new[new['movie_id'] == movie_id].index[0]

    # Get the similarity scores for the movie
    movie_scores = similarity_matrix[movie_index]

    # Sort the movies based on similarity scores
    sorted_indices = np.argsort(movie_scores)[::-1]
    sorted_scores = movie_scores[sorted_indices]
    sorted_titles = new.iloc[sorted_indices]['movie_id'].values

    # Select the top k recommendations
    top_recommendations = list(
        zip(sorted_titles[:top_k], sorted_scores[:top_k]))
    ids = [int(item[0]) for item in top_recommendations]

    return ids


# Generate recommendations for a movie
# recommendations_cosine = generate_recommendations(
#     similarity_matrix, 137106, 50)
recommendations_perfect = generate_recommendations(similarity_matrix, 1895, 50)
recommendations_normal = generate_recommendations(similarity_matrixn, 1895, 50)

num_users = 100
num_items = 500

# Define the interaction rate (e.g., 0.2 for 20% interactions per user)
interaction_rate = 0.2


app = Flask(__name__)


@app.route("/video/<int:movie_id>", methods=["GET"])
def hello_world(movie_id):
    print(generate_recommendations(similarity_matrix, movie_id, 5))
    return jsonify(generate_recommendations(similarity_matrix, movie_id, 5))
