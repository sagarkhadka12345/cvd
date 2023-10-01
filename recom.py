



from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np
import re
import ast
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from flask import Flask



movies = pd.read_csv('./videos.csv')
credits = pd.read_csv('./credits.csv')

movies.head(2)

movies.shape

credits.head()

movies = movies.merge(credits, on='title')

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

movies['tags'] = movies['overview'] + movies['subject'] + \
    movies['keywords'] + movies['cast'] + movies['crew']

new = movies
# new.head()

new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()



class CountVectorizer:
    def __init__(self, lowercase=True, token_pattern=r"(?u)\b\w\w+\b"):
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


cv = CountVectorizer()

new

vector = cv.fit_transform(new['tags']).toarray()
vector

vector.shape


# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit the vectorizer on the 'tags' column of the 'new' DataFrame
tfidf_matrix = vectorizer.fit_transform(new['tags'])

# Calculate the cosine similarity matrix
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to generate recommendations


def generate_recommendations(similarity_matrix, movie_id, top_k):
    # Find the index of the movie with the given title
    movie_index = new[new['movie_id'] == movie_id].index[0]
    print(movie_index)
    # Get the similarity scores for the movie
    movie_scores = similarity_matrix[movie_index]

    # Sort the movies based on similarity scores
    sorted_indices = np.argsort(movie_scores)[::-1]
    sorted_scores = movie_scores[sorted_indices]
    sorted_titles = new.iloc[sorted_indices]['movie_id'].values

    # Select the top k recommendations
    top_recommendations = list(
        zip(sorted_titles[:top_k], sorted_scores[:top_k]))

    return top_recommendations


# Generate recommendations for a movie
recommendations = generate_recommendations(similarity_matrix, 137106, 5)




app = Flask(__name__)

@app.route("/<int:movie_id>")
def hello_world(movie_id):
    return generate_recommendations(similarity_matrix, movie_id, 5)