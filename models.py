import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
class TextEmbedding(nn.Module):
    def __init__(self):
        super(TextEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

class GeoEmbedding(nn.Module):
    def __init__(self):
        super(GeoEmbedding, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def forward(self, geo_text):
        inputs = self.tokenizer(geo_text, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

class TimeEmbedding(nn.Module):
    def __init__(self):
        super(TimeEmbedding, self).__init__()
        self.fc = nn.Linear(1, 128)  # Simple linear time embedding

    def forward(self, timestamps):
        timestamps = timestamps.to(self.fc.weight.dtype)  # Ensure dtype consistency
        return self.fc(timestamps)

class MultiModalEmbedding(nn.Module):
    def __init__(self):
        super(MultiModalEmbedding, self).__init__()
        self.text_embed = TextEmbedding()
        self.geo_embed = GeoEmbedding()
        self.time_embed = TimeEmbedding()
        self.fc = nn.Linear(768 + 768 + 128, 512)  # Map combined embeddings to shared latent space

    def forward(self, text, geo_text, timestamps):
        text_vec = self.text_embed(text)
        geo_vec = self.geo_embed(geo_text)
        time_vec = self.time_embed(timestamps)
        combined = torch.cat((text_vec, geo_vec, time_vec), dim=1)
        return self.fc(combined)
class SemanticClassifier:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.categories = ['event description', 'city', 'timestamp']
        self.category_embeddings = self.model.encode(self.categories)

    def classify(self, token):
        # Timestamp placeholder
        if token.isdigit():
            if 1000000000 <= int(token) <= 2147483647:  # Unix timestamp range
                return 2  # timestamp
            else:
                return 0  # Default to event description for other numeric tokens
        token_embedding = self.model.encode([token])
        similarities = np.dot(token_embedding, self.category_embeddings.T)
        classification = np.argmax(similarities)
        if classification == 1 and any(char.isdigit() for char in token):
            return 0  # Prevent numeric tokens from being classified as location
        return classification
