

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


class CountVectorizerJaccard:
    def __init__(self, lowercase=True, token_pattern=r"(?u)\b\w\w+\b"):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.vocabulary = defaultdict(int)
        self.stop_words = set({"death", "foreign", "sextrafficking"})

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)

    def fit(self, raw_documents):
        for doc in raw_documents:
            tokens = self._tokenize(doc)
            for token in tokens:
                self.vocabulary[token] += 1.25

    def transform(self, raw_documents):
        rows, cols, data = [], [], []
        for i, doc in enumerate(raw_documents):
            tokens = self._tokenize(doc)
            for token in tokens:
                if token in self.vocabulary and token not in self.stop_words:
                    rows.append(i)
                    cols.append(self.vocabulary[token])
                    data.append(1.2)
        X = csr_matrix((data, (rows, cols)), shape=(
            len(raw_documents), len(self.vocabulary)))
        return X

    def _tokenize(self, text):
        if self.lowercase:
            text = text.lower()
        tokens = re.findall(self.token_pattern, text)
        return tokens


class CountVectorizerJaccardMultiColumn:
    def __init__(self, lowercase=True, token_pattern=r"(?u)\b\w\w+\b", column_weights={}):
        self.lowercase = lowercase
        self.token_pattern = token_pattern
        self.vocabulary = defaultdict(int)
        self.column_weights = column_weights if column_weights else {}
        self.stop_words = set({"death", "foreign", "sextrafficking"})

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def fit(self, df):
        for _, row in df.iterrows():
            for doc in row:
                tokens = self._tokenize(doc)
                for token in tokens:
                    if row in self.column_weights:
                        self.vocabulary[token] += self.column_weights[row]
                    else:
                        self.vocabulary[token] += 1

    def transform(self, df):
        rows, cols, data = [], [], []
        for i, row in df.iterrows():
            for col, weight in self.column_weights.items():
                text = row[col]
                tokens = self._tokenize(text)
                for token in tokens:
                    if token in self.vocabulary and token not in self.stop_words:
                        # Check if i is a valid row index
                        if i < len(df):
                            rows.append(i)
                            cols.append(self.vocabulary[token])
                            data.append(weight)
                        else:
                            print(f"Invalid row index: {i}")
        X = csr_matrix((data, (rows, cols)), shape=(
            len(df), len(self.vocabulary)))
        return X

    def _tokenize(self, text):
        if isinstance(text, str):
            if self.lowercase:
                text = text.lower()
            tokens = re.findall(self.token_pattern, text)
        elif isinstance(text, list):
            tokens = []
            for item in text:
                if isinstance(item, str):
                    if self.lowercase:
                        item = item.lower()
                    tokens.extend(re.findall(self.token_pattern, item))
        else:
            tokens = []
        return tokens
# Example usage:
# Create a DataFrame with multiple columns


# Example usage:
# Create a DataFrame with multiple columns


df = pd.DataFrame(movies)

# Define column weights (adjust as needed)
column_weights = {'cast': 1, 'crew': 1, 'tags': 1, 'keywords': 1, 'subject': 1}

# Initialize and use CountVectorizerJaccardMultiColumn
vectorizer = CountVectorizerJaccardMultiColumn()
X = vectorizer.fit_transform(df)


cv = CountVectorizer()
cvd = CountVectorizerJaccard()




# Fit the vectorizer on the 'tags' column of the 'new' DataFrame
tfidf_matrix = cv.fit_transform(new['tags'])
tfidf_matrixj = cv.fit_transform(
    new['tags'])


# Compute Pearson similarity matrix

similarity_matrix = cosine_similarity(X, X)


num_docs, num_terms = tfidf_matrix.shape
document_sets = []

for i in range(num_docs):
    # Get the non-zero indices (terms with non-zero TF-IDF values)
    non_zero_indices = np.nonzero(tfidf_matrix[i])[1]
    # Create a set of terms for the document
    document_set = set(non_zero_indices)
    document_sets.append(document_set)


 # Since Jaccard similarity is symmetric
def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union

# Calculate Jaccard similarities between documents
num_docs = len(document_sets)
jaccard_similarities = np.zeros((num_docs, num_docs))

for i in range(num_docs):
    for j in range(i, num_docs):
        jaccard_similarities[i, j] = jaccard_similarity(document_sets[i], document_sets[j])
        jaccard_similarities[j, i] = jaccard_similarities[i, j]
# Print the Jaccard similarities


# Calculate the cosine similarity matrix
similarity_matrix_p = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Subtract the mean from each document's TF-IDF values
centered_tfidf = tfidf_matrix - mean_tfidf

# Calculate the Pearson Correlation Coefficient (PCC) similarity matrix

print("Shape of tfidf_matrix:", tfidf_matrix.shape)
print("Shape of mean_tfidf:", mean_tfidf.shape)
print("Data type of tfidf_matrix:", tfidf_matrix.dtype)
print("Data type of mean_tfidf:", mean_tfidf.dtype)
# Subtract the mean from each document's TF-IDF values


# Calculate the Pearson Correlation Coefficient (PCC) similarity matrix
pcc_similarity_matrix = np.corrcoef(centered_tfidf)
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
recommendations_cosine = generate_recommendations(
    similarity_matrix, 137106, 50)
recommendations_p = generate_recommendations(similarity_matrix_p, 137106, 50)


# Convert your arrays to NumPy arrays
recommendations_cosine = np.array(recommendations_cosine)

recommendations_p = np.array(recommendations_p)


# Calculate Pearson correlation coefficient
pearson_similarity = np.corrcoef(
    recommendations_cosine, recommendations_p)[0, 1]

print("Pearson Correlation Coefficient:", pearson_similarity)

# Define a list of items (e.g., item IDs or item features)
items = recommendations_cosine
print(items)
# Define the number of synthetic users and the desired interaction rate
# num_users = 100
# interaction_rate = .9  # Adjust as needed (percentage of interactions)

# # Create an empty dictionary to store synthetic interactions
# synthetic_interactions = {}

# # Generate synthetic interactions for each user
# for user_id in range(1, num_users + 1):
#     # Randomly choose the number of interactions for this user
#     num_interactions = int(len(items) * interaction_rate)
#     print(num_interactions , items)
#     # Randomly sample items for interactions
#     user_interactions = random.sample(items, num_interactions)
    
#     # Store the interactions for this user in the dictionary
#     synthetic_interactions[f"user{user_id}"] = user_interactions

# # Print the synthetic interactions
# for user_id, interactions in synthetic_interactions.items():
#     print(f"User {user_id} interactions: {interactions}")
    
num_users = 100
num_items = 500

# Define the interaction rate (e.g., 0.2 for 20% interactions per user)
interaction_rate = 0.2

# Initialize a list to store synthetic interactions
synthetic_interactions = []

# Generate synthetic interactions for each user
for user_id in range(1, num_users + 1):
    # Randomly choose the number of interactions for this user
    num_interactions = int(num_items * interaction_rate)
    
    # Randomly sample items for interactions (using item IDs)
    user_interactions = random.sample(range(1, num_items + 1), num_interactions)
    
    # Append the user's interactions to the list
    synthetic_interactions.append(user_interactions)

# Print the synthetic interactions
for user_id, interactions in enumerate(synthetic_interactions, start=1):
    print(f"User {user_id} interactions: {interactions}")


# Assuming 'synthetic_interactions' is a dictionary of synthetic user interactions
# 'recommended_items' is a dictionary of recommended items where keys are user IDs and values are lists of recommended item IDs.

total_average_precision = 0
total_users = len(synthetic_interactions)

# Iterate through each user (assuming each inner list represents a user's interactions)
for i in range(total_users):
    actual_interactions_list = synthetic_interactions[i]
    recommended_video_ids = recommendations_cosine[i]

    # Initialize variables for this user's AP calculation
    num_relevant_recommendations = 0
    precision_sum = 0

    # Calculate Average Precision (AP) for this user
    for j, recommended_video_id in enumerate(recommended_video_ids):
        if recommended_video_id in actual_interactions_list:
            num_relevant_recommendations += 1
            precision = num_relevant_recommendations / (j + 1)
            precision_sum += precision

    # Calculate the Average Precision (AP) for this user
    if num_relevant_recommendations > 0:
        average_precision = precision_sum / num_relevant_recommendations
        total_average_precision += average_precision

# Calculate the Mean Average Precision (MAP) for the entire dataset
mean_average_precision = total_average_precision / total_users

# Print the Mean Average Precision (MAP)
print("Mean Average Precision (MAP):", mean_average_precision)
accuracy = np.corrcoef(recommendations, recommendations_cosine)[0, 1]

print(f"Accuracy: {accuracy:.2f}%")

app = Flask(__name__)


@app.route("/video/<int:movie_id>", methods=["GET"])
def hello_world(movie_id):
    print(generate_recommendations(similarity_matrix, movie_id, 5))
    return jsonify(generate_recommendations(similarity_matrix, movie_id, 5))
