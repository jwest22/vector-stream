import numpy as np
import networkx as nx

def get_node_features(G):
    # Example of extracting simple features (like degrees) for demonstration purposes
    features = {node: [data['type'] == 'customer', data['type'] == 'product'] for node, data in G.nodes(data=True)}
    return features

def encode_features(features):
    # Convert dictionary of features into numpy array or similar structure for node2vec
    nodes = list(features.keys())
    encoded_features = np.array([features[node] for node in nodes])
    return encoded_features
