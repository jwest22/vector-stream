# main.py

import torch
import faiss
import numpy as np
import os
from models import MultiModalEmbedding

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = MultiModalEmbedding()

events = [
    {"description": "Event description 1", "location": "New York City", "timestamp": 1625097600},
    {"description": "Event description 2", "location": "Los Angeles", "timestamp": 1625184000},
    {"description": "Event description 3", "location": "New York City", "timestamp": 1625270400},
]

text_embeddings = []
geo_embeddings = []
time_embeddings = []
combined_embeddings = []

for event in events:
    text_embedding = model.text_embed([event["description"]])
    geo_embedding = model.geo_embed([event["location"]])
    time_embedding = model.time_embed(torch.tensor([[event["timestamp"]]], dtype=torch.float32))
    
    combined_embedding = model.fc(torch.cat((text_embedding, geo_embedding, time_embedding), dim=1))
    
    text_embeddings.append(text_embedding.detach().numpy())
    geo_embeddings.append(geo_embedding.detach().numpy())
    time_embeddings.append(time_embedding.detach().numpy())
    combined_embeddings.append(combined_embedding.detach().numpy())

text_embeddings_np = np.vstack(text_embeddings)
geo_embeddings_np = np.vstack(geo_embeddings)
time_embeddings_np = np.vstack(time_embeddings)
combined_embeddings_np = np.vstack(combined_embeddings)

text_dimension = text_embeddings_np.shape[1]
geo_dimension = geo_embeddings_np.shape[1]
time_dimension = time_embeddings_np.shape[1]
combined_dimension = combined_embeddings_np.shape[1]

text_index = faiss.IndexFlatL2(text_dimension)
geo_index = faiss.IndexFlatL2(geo_dimension)
time_index = faiss.IndexFlatL2(time_dimension)
combined_index = faiss.IndexFlatL2(combined_dimension)

text_index.add(text_embeddings_np)
geo_index.add(geo_embeddings_np)
time_index.add(time_embeddings_np)
combined_index.add(combined_embeddings_np)

query_geo_text = ["New York City"]
query_geo_embedding = model.geo_embed(query_geo_text).detach().numpy()
D_geo, I_geo = geo_index.search(query_geo_embedding, k=len(events))

print("Events in New York City:")
for distance, idx in zip(D_geo[0], I_geo[0]):
    print(f"Distance: {distance}, Event: {events[idx]}")

query_time = torch.tensor([[1625184000]], dtype=torch.float32)
query_time_embedding = model.time_embed(query_time).detach().numpy()
D_time, I_time = time_index.search(query_time_embedding, k=len(events))

print("Events with similar timestamps:")
for distance, idx in zip(D_time[0], I_time[0]):
    print(f"Distance: {distance}, Event: {events[idx]}")

query_text_description = ["Event description 1"]
query_text_embedding = model.text_embed(query_text_description).detach().numpy()
D_text, I_text = text_index.search(query_text_embedding, k=len(events))

print("Events with similar descriptions:")
for distance, idx in zip(D_text[0], I_text[0]):
    print(f"Distance: {distance}, Event: {events[idx]}")
    
query_text = ["Event description 1"]
query_geo_text = ["New York City"]
query_time = torch.tensor([[1625097600]], dtype=torch.float32)

query_text_embedding = model.text_embed(query_text)
query_geo_embedding = model.geo_embed(query_geo_text)
query_time_embedding = model.time_embed(query_time)

combined_query_embedding = model.fc(torch.cat((query_text_embedding, query_geo_embedding, query_time_embedding), dim=1)).detach().numpy()
D_combined, I_combined = combined_index.search(combined_query_embedding, k=len(events))

print("Combined query results:")
for distance, idx in zip(D_combined[0], I_combined[0]):
    print(f"Distance: {distance}, Event: {events[idx]}")
