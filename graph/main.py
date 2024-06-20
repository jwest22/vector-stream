import duckdb
import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
from sentence_transformers import SentenceTransformer

# Transformer-Based Graph Embedding for Node Clustering

# Load data
accounts_df = pd.read_csv('demo_data/sfdc_account.csv')
opportunities_df = pd.read_csv('demo_data/sfdc_opportunity.csv')

# Ensure date columns are in correct format
accounts_df['created_date'] = pd.to_datetime(accounts_df['created_date'], errors='coerce')
opportunities_df['created_date'] = pd.to_datetime(opportunities_df['created_date'], errors='coerce')
opportunities_df['close_date'] = pd.to_datetime(opportunities_df['close_date'], errors='coerce')

# Connect to DuckDB
con = duckdb.connect()

# Register DataFrames
con.register('accounts_df', accounts_df)
con.register('opportunities_df', opportunities_df)

# Create tables in DuckDB
con.execute("CREATE OR REPLACE TABLE accounts AS SELECT * FROM accounts_df")
con.execute("CREATE OR REPLACE TABLE opportunities AS SELECT * FROM opportunities_df")

# Query to get result_df with all necessary joins and columns
query = """
SELECT 
    a.id AS account_id, 
    a.name AS account_name, 
    a.type AS account_type,
    a.industry AS account_industry,
    a.billingcountry AS account_country_name,
    a.created_date AS account_created_date,
    o.id AS opportunity_id, 
    o.name AS opportunity_name,
    o.stage_name AS opportunity_stage_name,
    o.amount AS opportunity_amount,
    o.probability AS opportunity_probability,
    o.close_date AS opportunity_close_date,
    o.is_closed,
    o.is_won,
    o.created_date AS opportunity_created_date
FROM accounts a
LEFT JOIN opportunities o
ON a.id = o.account_id
"""

result_df = con.execute(query).fetchdf()

# Aggregate opportunity data for each account
opportunity_agg = result_df.groupby('account_id').agg({
    'opportunity_id': 'count',
    'opportunity_amount': 'sum',
    'opportunity_probability': 'mean',
    'is_won': 'mean'
}).reset_index()
opportunity_agg.rename(columns={
    'opportunity_id': 'opportunity_count',
    'opportunity_amount': 'total_opportunity_amount',
    'opportunity_probability': 'avg_opportunity_probability',
    'is_won': 'won_proportion'
}, inplace=True)

# Merge the aggregated data back to the result_df
result_df = result_df.drop_duplicates(subset=['account_id']).merge(opportunity_agg, on='account_id', how='left')

# Create an undirected graph
G = nx.Graph()

# Store labels for visualization
labels_dict = {}

# Add nodes from accounts with aggregated opportunity attributes
for _, row in result_df.iterrows():
    if not G.has_node(row['account_id']):
        G.add_node(row['account_id'], 
                   type='account', 
                   name=row['account_name'], 
                   account_type=row['account_type'], 
                   industry=row['account_industry'], 
                   country=row['account_country_name'], 
                   created_date=row['account_created_date'],
                   opportunity_count=row['opportunity_count'],
                   total_opportunity_amount=row['total_opportunity_amount'],
                   avg_opportunity_probability=row['avg_opportunity_probability'],
                   won_proportion=row['won_proportion'])
        labels_dict[row['account_id']] = row['account_name']

# Using DuckDB to add edges based on the same industry
edges_query = """
SELECT a1.id AS account1, a2.id AS account2
FROM accounts a1
JOIN accounts a2
ON a1.industry = a2.industry
AND a1.id != a2.id
"""

industry_edges_df = con.execute(edges_query).fetchdf()

# Add edges for same industry
for _, row in industry_edges_df.iterrows():
    G.add_edge(row['account1'], row['account2'])

# Using DuckDB to add edges based on the same month and year for created_date
edges_query_created_date = """
SELECT a1.id AS account1, a2.id AS account2
FROM accounts a1
JOIN accounts a2
ON DATE_TRUNC('month', CAST(a1.created_date AS TIMESTAMP)) = DATE_TRUNC('month', CAST(a2.created_date AS TIMESTAMP))
AND a1.id != a2.id
"""

created_date_edges_df = con.execute(edges_query_created_date).fetchdf()

# Add edges for same created date
for _, row in created_date_edges_df.iterrows():
    G.add_edge(row['account1'], row['account2'])

# Using DuckDB to add edges based on the same month and year for close_date
edges_query_close_date = """
SELECT o1.account_id AS account1, o2.account_id AS account2
FROM opportunities o1
JOIN opportunities o2
ON DATE_TRUNC('month', CAST(o1.close_date AS TIMESTAMP)) = DATE_TRUNC('month', CAST(o2.close_date AS TIMESTAMP))
AND o1.account_id != o2.account_id
"""

close_date_edges_df = con.execute(edges_query_close_date).fetchdf()

# Add edges for same close date
for _, row in close_date_edges_df.iterrows():
    G.add_edge(row['account1'], row['account2'])

# Extract attributes as features
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
            data.get('won_proportion', 0.0)
        ])
    return account_features

# Convert features using Sentence-BERT for textual data
def encode_features(features):
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    encoded_features = []
    for feature_set in features:
        text_features = [str(feature) for feature in feature_set[:-4]]  # all but the last four features
        numerical_features = feature_set[-4:]  # the last four features
        text_embedding = model.encode(text_features)
        numerical_features = np.array(numerical_features).reshape(-1, 1)
        combined_features = np.concatenate((text_embedding, numerical_features), axis=None)
        encoded_features.append(combined_features)
    return np.array(encoded_features)

# Get and encode features
features = get_node_features(G)
features = encode_features(features)

# Initialize Node2Vec
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)

# Fit the model
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Save model
model.save('node2vec.model')

# Load model (for demonstration)
model = Word2Vec.load('node2vec.model')

# Get embeddings for all nodes
embeddings = np.array([model.wv[str(node)] for node in G.nodes()])

# Clustering using K-means
kmeans = KMeans(n_clusters=5, random_state=0).fit(embeddings)
labels = kmeans.labels_

# Assign cluster labels to nodes
for i, node in enumerate(G.nodes()):
    G.nodes[node]['cluster'] = labels[i]

# Visualize clusters using t-SNE
n_samples = embeddings.shape[0]
perplexity = min(30, n_samples - 1)
tsne = TSNE(n_components=2, perplexity=perplexity)
embeddings_2d = tsne.fit_transform(embeddings)

# Get colors based on won_proportion for accounts
colors = [G.nodes[node]['won_proportion'] for node in G.nodes()]

plt.figure(figsize=(12, 12))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='viridis')
for i, node in enumerate(G.nodes()):
    plt.annotate(labels_dict[node], (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.colorbar(label='Won Proportion')
plt.show()
