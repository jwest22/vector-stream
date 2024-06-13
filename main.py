import streamlit as st
import torch
import faiss
import numpy as np
import os
import warnings
from models import MultiModalEmbedding, SemanticClassifier
import helpers
import pandas as pd

# Suppress specific FutureWarning from Hugging Face
warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Initialize models
model = MultiModalEmbedding()
classifier = SemanticClassifier()

# Sample events
events = [
    {"description": "Event description 1", "location": "New York City", "timestamp": 1625097600},
    {"description": "Event description 2", "location": "Los Angeles", "timestamp": 1625184000},
    {"description": "Event description 3", "location": "New York City", "timestamp": 1625270400},
]

# Embed events
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

# Convert to numpy arrays
text_embeddings_np = np.vstack(text_embeddings)
geo_embeddings_np = np.vstack(geo_embeddings)
time_embeddings_np = np.vstack(time_embeddings)
combined_embeddings_np = np.vstack(combined_embeddings)

# Create FAISS indices
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

# Streamlit interface
st.title("Vector Stream")

user_query = st.text_input("Enter your query:", "Event description 1 New York City 1625097600")

if st.button("Run Query"):
    query_text, query_geo, query_time, token_classification = helpers.parse_and_classify_query(user_query)

    # Combine tokens by classification
    classification_dict = {}
    for token, classification in token_classification:
        if classification not in classification_dict:
            classification_dict[classification] = []
        classification_dict[classification].append(token)
    
    combined_classification = {classification: " ".join(tokens) for classification, tokens in classification_dict.items()}
    combined_classification_df = pd.DataFrame(list(combined_classification.items()), columns=["Classification", "Tokens"])
    st.write("**Token Classification:**")
    st.dataframe(combined_classification_df)

    query_text_embedding = model.text_embed([query_text]) if query_text else torch.zeros((1, 768))
    query_geo_embedding = model.geo_embed([query_geo]) if query_geo else torch.zeros((1, 768))
    query_time_embedding = model.time_embed(torch.tensor([[int(query_time[0])]], dtype=torch.float32)) if query_time else torch.zeros((1, 128))

    combined_query_embedding = model.fc(torch.cat((query_text_embedding, query_geo_embedding, query_time_embedding), dim=1)).detach().numpy()
    D_combined, I_combined = combined_index.search(combined_query_embedding, k=len(events))

    results = []
    for distance, idx in zip(D_combined[0], I_combined[0]):
        event = events[idx]
        event['distance'] = distance
        event['timestamp'] = pd.to_datetime(event['timestamp'], unit='s')  # Convert to timestamp
        results.append(event)
    
    results_df = pd.DataFrame(results)
    st.write("**Combined Query Results:**")
    st.dataframe(results_df)
