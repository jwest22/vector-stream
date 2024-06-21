import numpy as np
from sentence_transformers import SentenceTransformer

def get_node_features(G):
    account_features = []
    for node, data in G.nodes(data=True):
        account_features.append([
            data.get('account_type', ''), 
            data.get('industry', ''), 
            data.get('country', ''), 
            data.get('created_date', ''), 
            data.get('opportunity_count', 0), 
            data.get('total_opportunity_amount', 0.0), 
            data.get('avg_opportunity_probability', 0.0), 
            data.get('recent_is_won', 0.0)
        ])
    return account_features

def encode_features(features):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    encoded_features = []
    for feature_set in features:
        text_features = [str(feature) for feature in feature_set[:-4]]
        numerical_features = feature_set[-4:]
        text_embedding = model.encode(text_features)
        numerical_features = np.array(numerical_features).reshape(-1, 1)
        combined_features = np.concatenate((text_embedding, numerical_features), axis=None)
        encoded_features.append(combined_features)
    return np.array(encoded_features)
