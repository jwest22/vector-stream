import pandas as pd
import networkx as nx
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt

# Define the folder containing the Retailrocket dataset
data_folder = 'retailrocket_data'

# Load the data
events = pd.read_csv(os.path.join(data_folder, 'events.csv'))
item_properties_part1 = pd.read_csv(os.path.join(data_folder, 'item_properties_part1.csv'))
item_properties_part2 = pd.read_csv(os.path.join(data_folder, 'item_properties_part2.csv'))

# Combine item properties parts
item_properties = pd.concat([item_properties_part1, item_properties_part2])

# Example: Display the first few rows of each DataFrame
print("Events:\n", events.head())
print("Item Properties:\n", item_properties.head())

# Create a graph
G = nx.Graph()

# Add items (products) as nodes
for item_id in item_properties['itemid'].unique():
    G.add_node(item_id, type='item')

# Add users as nodes and create edges based on interactions
for _, row in events.iterrows():
    user_id = row['visitorid']
    item_id = row['itemid']
    interaction_type = row['event']  # 'view', 'addtocart', 'transaction'
    
    # Add user node if it doesn't exist
    if not G.has_node(user_id):
        G.add_node(user_id, type='user')
    
    # Add an edge between the user and the item
    G.add_edge(user_id, item_id, interaction=interaction_type)

# Ensure all nodes have a 'type' attribute
for node in G.nodes:
    if 'type' not in G.nodes[node]:
        G.nodes[node]['type'] = 'unknown'

# Convert NetworkX graph to PyTorch Geometric Data
def from_networkx_to_torchgeo(G):
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    
    # Create simple features: 1 for users, 0 for items
    num_nodes = len(G.nodes)
    features = np.zeros((num_nodes, 1))
    for idx, node in enumerate(G.nodes):
        features[idx] = 1 if G.nodes[node]['type'] == 'user' else 0
    
    x = torch.tensor(features, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    return data

data = from_networkx_to_torchgeo(G)

# Create labels: 1 for transaction, 0 otherwise
labels = np.zeros(len(G.nodes))
for edge in G.edges(data=True):
    user, item = edge[0], edge[1]
    if edge[2]['interaction'] == 'transaction':
        labels[user] = 1

# Convert labels to PyTorch tensors
data.y = torch.tensor(labels, dtype=torch.long)

# Create train/test masks
data.train_mask = torch.rand(len(G.nodes)) < 0.8  # 80% training data
data.test_mask = ~data.train_mask  # 20% test data

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Initialize and train the GCN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=1, hidden_channels=16, out_channels=2).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

model.eval()
_, pred = model(data).max(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

# Optional: Visualize a small subgraph
def visualize_subgraph(G, num_nodes=100):
    subgraph = G.subgraph(list(G.nodes)[:num_nodes])
    pos = nx.spring_layout(subgraph)

    plt.figure(figsize=(12, 12))
    nx.draw(subgraph, pos, with_labels=True, node_size=50, font_size=8)
    plt.show()

# Visualize the first 100 nodes
visualize_subgraph(G, num_nodes=100)
