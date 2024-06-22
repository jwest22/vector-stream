import numpy as np
from node2vec import Node2Vec
from gensim.models import Word2Vec
from sklearn.cluster import KMeans

def generate_node2vec_embeddings(G, features=None):
    if features is not None:
        node2vec = Node2Vec(G, dimensions=features.shape[1], walk_length=30, num_walks=200, workers=4, p=1, q=1, weight_key='weight')
    else:
        node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4, weight_key='weight')
    model = node2vec.fit(window=10, min_count=1, batch_words=4)
    model.save('node2vec.model')
    return model

def load_node2vec_model():
    model = Word2Vec.load('node2vec.model')
    return model

def get_embeddings(model, G):
    embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    return embeddings

def perform_clustering(embeddings, G):
    kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
    labels = kmeans.labels_
    for i, node in enumerate(G.nodes()):
        G.nodes[node]['cluster'] = labels[i]
