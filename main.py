import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import faiss
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Text Embedding Model
class TextEmbedding(nn.Module):
    def __init__(self):
        super(TextEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

# Geolocation (text) Embedding Model
class GeoEmbedding(nn.Module):
    def __init__(self):
        super(GeoEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, geo_text):
        inputs = self.tokenizer(geo_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

# Timestamp Embedding Model
class TimeEmbedding(nn.Module):
    def __init__(self):
        super(TimeEmbedding, self).__init__()
        self.fc = nn.Linear(1, 128)  # Simple linear time embedding

    def forward(self, timestamps):
        timestamps = timestamps.to(self.fc.weight.dtype)  # Ensure dtype consistency
        return self.fc(timestamps)

# Example
text_model = TextEmbedding()
geo_model = GeoEmbedding()
time_model = TimeEmbedding()

events = [
    {"description": "Event description 1", "location": "New York City", "timestamp": 1625097600},
    {"description": "Event description 2", "location": "Los Angeles", "timestamp": 1625184000},
    {"description": "Event description 3", "location": "New York City", "timestamp": 1625270400},
]

text_embeddings = []
geo_embeddings = []
time_embeddings = []

for event in events:
    text_embedding = text_model([event["description"]])
    geo_embedding = geo_model([event["location"]])
    time_embedding = time_model(torch.tensor([[event["timestamp"]]], dtype=torch.float32))
    
    text_embeddings.append(text_embedding.detach().numpy())
    geo_embeddings.append(geo_embedding.detach().numpy())
    time_embeddings.append(time_embedding.detach().numpy())

text_embeddings_np = np.vstack(text_embeddings)
geo_embeddings_np = np.vstack(geo_embeddings)
time_embeddings_np = np.vstack(time_embeddings)

geo_dimension = geo_embeddings_np.shape[1]
geo_index = faiss.IndexFlatL2(geo_dimension)

geo_index.add(geo_embeddings_np)

# Generate embeddings for location query
query_geo_text = ["New York City"]
query_geo_embedding = geo_model(query_geo_text)
query_geo_embedding_np = query_geo_embedding.detach().numpy()

# Search geo index
D_geo, I_geo = geo_index.search(query_geo_embedding_np, k=len(events))

# Print all events in New York City & their distances
print("Events in New York City:")
for distance, idx in zip(D_geo[0], I_geo[0]):
    print(f"Distance: {distance}, Event: {events[idx]}")
