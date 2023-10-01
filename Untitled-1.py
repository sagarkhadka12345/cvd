# %%


from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from collections import defaultdict
import numpy as np
import re
import ast
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %%
movies = pd.read_csv('./archive/tmdb_5000_movies.csv')
credits = pd.read_csv('./archive/tmdb_5000_credits.csv')

# %%
movies.head(2)

# %%
movies.shape

# %%
credits.head()

# %%
movies = movies.merge(credits, on='title')

# %%
movies.head()
# budget
# homepage
# id
# original_language
# original_title
# popularity
# production_comapny
# production_countries
# release-date(not sure)

# %%
movies = movies[['movie_id', 'title', 'overview',
                 'genres', 'keywords', 'cast', 'crew']]

# %%
movies.head()

# %%

# %%


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


# %%
movies.dropna(inplace=True)

# %%
movies['genres'] = movies['genres'].apply(convert)
movies.head()

# %%
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()

# %%
ast.literal_eval(
    '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')

# %%


def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L


# %%
movies['cast'] = movies['cast'].apply(convert)
movies.head()

# %%
movies['cast'] = movies['cast'].apply(lambda x: x[0:3])

# %%


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L


# %%
movies['crew'] = movies['crew'].apply(fetch_director)

# %%
# movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample(5)

# %%


def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1


# %%
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# %%
movies.head()

# %%
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# %%
movies['tags'] = movies['overview'] + movies['genres'] + \
    movies['keywords'] + movies['cast'] + movies['crew']

# %%
new = movies
# new.head()

# %%
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
new.head()

# %%


class CustomCountVectorizer:
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


cv = CustomCountVectorizer()

# %%
new

# %%
vector = cv.fit_transform(new['tags']).toarray()
vector

# %%
vector.shape

# %%

# %%


def custom_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 != 0 and norm2 != 0:
        similarity = dot_product / (norm1 * norm2)
    else:
        similarity = 0

    return similarity


# Get the number of documents and terms in the count matrix
num_docs, num_terms = vector.shape

# Initialize the similarity matrix with zeros
similarity_matrix = np.zeros((num_docs, num_docs))

# Calculate the custom cosine similarity between each pair of documents
for i in range(num_docs):
    for j in range(i, num_docs):
        similarity = custom_cosine_similarity(vector[i], vector[j])
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity


# %%


# %%
similarity_matrix

# %%


# %%
new[new['title'] == 'The Lego Movie'].index[0]
new

# %%


def generate_recommendations(similarity_matrix, user_id, top_k):
    # Get the similarity scores for the user_id
    user_row = similarity_matrix[user_id, :]

    # Sort the similarity scores in descending order along with their corresponding IDs
    sorted_indices = np.argsort(user_row)[::-1]
    sorted_scores = user_row[sorted_indices]
    sorted_ids = sorted_indices.tolist()

    # Select the top k recommendations
    top_recommendations = sorted_ids[:top_k]

    return top_recommendations


# %%
generate_recommendations(similarity_matrix, 'Newlyweds', 5)

# %%
