

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
# movies = pd.read_csv('./videos.csv')
# credits = pd.read_csv('./credits.csv')



host = 'localhost'
user = 'root'
password = ''
database = 'cvd'
connection = pymysql.connect(
    host=host, user=user, password=password, database=database)
query = 'SELECT * FROM `videos` ORDER BY `release_date` DESC'
querycredits = 'SELECT * FROM credits'


movies = pd.read_sql(query, connection)
movies.head(2)

movies.shape
movies.head(2)

movies.shape




movies.head()

credits = pd.read_sql(querycredits, connection)

print(movies)

movies = movies.merge(credits, on='title', how='left')
movies.fillna('', inplace=True)

print(movies["movie_id"])
movies['movie_id'] = movies['id']
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


# movies['crew'] = movies['crew'].apply(fetch_director)

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




num_users = 100
num_items = 500

mv = movies
mv


movies['overview']= movies['overview'].apply(lambda x: " ".join(x))
movies['subject']= movies['subject'].apply(lambda x: " ".join(x))
movies['keywords']= movies['keywords'].apply(lambda x: " ".join(x))
movies['cast']= movies['cast'].apply(lambda x: " ".join(x))
movies['crew']= movies['crew'].apply(lambda x: " ".join(x))



new = movies
# new.head()





new.head()

# %%


from collections import defaultdict
import re
from scipy.sparse import csr_matrix




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






# %%
df = pd.DataFrame(mv)


column_weights_main = {'cast': 1.197, 'crew': 1.3, 'tags': 1.2, 'keywords': 1.1, 'subject': 1, "overview":1}




cvn = CountVectorizerAllColumns(column_weights=column_weights_main)

# 

print(mv)

# Fit the vectorizer on the 'tags' column of the 'new' DataFrame

tfidf_matrix = cvn.fit_transform(mv)



similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)


# num_docs, num_terms = tfidf_matrix.shape
document_sets = []


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

print(similarity_matrix)

recommendations_n = generate_recommendations(similarity_matrix, 25397, 50)
print(recommendations_n)
recom = pd.read_csv("./recommendationsnew.csv")

# %%
# def generate_recommendations(similarity_matrix, movie_id, top_k):
#     # Find the index of the movie with the given title
#     movie_index = recom[recom['movie_id'] == movie_id].index[0]

#     # Get the similarity scores for the movie
#     movie_scores = similarity_matrix[movie_index]

#     # Sort the movies based on similarity scores
#     sorted_indices = np.argsort(movie_scores)[::-1]
#     sorted_scores = movie_scores[sorted_indices]
#     sorted_titles = recom.iloc[sorted_indices]['movie_id'].values

#     # Select the top k recommendations
#     top_recommendations = list(
#         zip(sorted_titles[:top_k], sorted_scores[:top_k]))
#     ids = [int(item[0]) for item in top_recommendations]

#     return ids


app = Flask(__name__)


@app.route("/video/<int:movie_id>", methods=["GET"])
def hello_world(movie_id):
    try:
        print(movie_id)
        return jsonify(generate_recommendations(similarity_matrix, movie_id, 5))
    except Exception as e:
        print(e)

