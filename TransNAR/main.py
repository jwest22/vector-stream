import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from transformers import AutoTokenizer

# Check for CPU usage since ROCm is not available
device = torch.device("cpu")
print("Using device:", device)

class TransNAR(nn.Module):
    def __init__(self, d_model, d_k, num_layers, nhead, projected_dim, vocab_size):
        super(TransNAR, self).__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.num_layers = num_layers
        self.projected_dim = projected_dim
        self.vocab_size = vocab_size

        # Embedding layer for text input
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Linear projection layer
        self.projection = nn.Linear(d_model, projected_dim)

        # Transformer components
        self.text_transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(projected_dim, nhead=nhead, dim_feedforward=2048) for _ in range(num_layers)
        ])
        
        # NAR components
        self.nar_layers = nn.ModuleList([
            MaxMPNN(projected_dim) for _ in range(num_layers)
        ])
        
        # Cross-attention components
        self.query_transform = nn.Linear(projected_dim, d_k)
        self.key_transform = nn.Linear(projected_dim, d_k)
        self.value_transform = nn.Linear(projected_dim, projected_dim)
        self.final_ffn = nn.Linear(projected_dim, projected_dim)

        # Generation head
        self.generation_head = nn.Linear(projected_dim, vocab_size)
        
    def forward(self, text_input_ids, graph_input):
        # Convert token IDs to embeddings
        text_input = self.embedding(text_input_ids)
        
        # Project the inputs
        text_input = self.projection(text_input)
        graph_input = self.projection(graph_input)

        T = text_input
        G = graph_input

        for t in range(self.num_layers):
            # Transformer layer
            T = self.text_transformer_layers[t](T)

            # NAR layer
            G = self.nar_layers[t](G)

            # Cross-attention
            Q = self.query_transform(T)
            K = self.key_transform(G)
            V = self.value_transform(G)
            
            attn_weights = F.softmax(Q.bmm(K.transpose(1, 2)) / self.d_k**0.5, dim=-1)
            cross_attn_output = attn_weights.bmm(V)
            T = self.final_ffn(cross_attn_output)

        # Generation
        logits = self.generation_head(T)
        return logits

    def generate_text(self, initial_text, graph_input, tokenizer, max_length=20):
        self.eval()
        with torch.no_grad():
            # Encode the initial text
            input_ids = tokenizer.encode(initial_text, return_tensors='pt').to(graph_input.device)
            
            generated_ids = input_ids[0].tolist()

            for _ in range(max_length):
                # Forward pass
                logits = self(input_ids, graph_input)
                next_token_id = logits[:, -1, :].argmax(dim=-1).item()
                
                if next_token_id == tokenizer.eos_token_id:
                    break
                print(next_token_id)  
                generated_ids.append(next_token_id)
                input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=input_ids.device)], dim=1)

            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            return generated_text

class MaxMPNN(nn.Module):
    def __init__(self, d_model):
        super(MaxMPNN, self).__init__()
        self.d_model = d_model
        self.message_func = nn.Linear(d_model * 2, d_model)
        self.update_func = nn.Linear(d_model * 2, d_model)

    def forward(self, G):
        batch_size, num_nodes, _ = G.size()
        G_new = torch.zeros_like(G)
        
        for u in range(num_nodes):
            neighbor_messages = []
            for v in range(num_nodes):
                if u != v:
                    message = self.message_func(torch.cat([G[:, u, :], G[:, v, :]], dim=-1))
                    neighbor_messages.append(message)
            
            if neighbor_messages:
                neighbor_messages = torch.stack(neighbor_messages, dim=1)
                max_message = torch.max(neighbor_messages, dim=1).values
                G_new[:, u, :] = self.update_func(torch.cat([G[:, u, :], max_message], dim=-1))
            else:
                G_new[:, u, :] = G[:, u, :]
        
        return G_new

# Measure time for dataset loading
start_time = time.time()

# Use a temporary directory to store the dataset
tmp_dir = os.path.join(os.getcwd(), 'tmp')
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)
dataset_path = os.path.join(tmp_dir, 'Cora')

# Load the Cora dataset
dataset = Planetoid(root=dataset_path, name='Cora')
data = dataset[0]

# Measure time for dataset loading
dataset_loading_time = time.time() - start_time
print(f"Dataset loading time: {dataset_loading_time:.2f} seconds")

# Convert to dense format for compatibility with the current model
adjacency_matrix = torch.eye(data.num_nodes)
adjacency_matrix[data.edge_index[0], data.edge_index[1]] = 1
graph_input = data.x.unsqueeze(0).to(device)  # Add batch dimension and move to CPU

# Example text input
batch_size = 1
num_tokens = 5  # Initial token length
d_model = data.num_node_features  # 1433 for Cora
d_k = 32
num_layers = 2
nhead = 8
projected_dim = 512  # Must be divisible by nhead

# Use the Hugging Face tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
vocab_size = tokenizer.vocab_size

# Measure time for model initialization
start_time = time.time()

# Initialize the model
model = TransNAR(d_model, d_k, num_layers, nhead, projected_dim, vocab_size).to(device)

model_initialization_time = time.time() - start_time
print(f"Model initialization time: {model_initialization_time:.2f} seconds")

# Measure time for text generation
start_time = time.time()

# Example of text generation
initial_text = "this is"

# Generate text
generated_text = model.generate_text(initial_text, graph_input, tokenizer)
text_generation_time = time.time() - start_time
print("Generated text:", generated_text)
print(f"Text generation time: {text_generation_time:.2f} seconds")

total_time = dataset_loading_time + model_initialization_time + text_generation_time
print(f"Total execution time: {total_time:.2f} seconds")
