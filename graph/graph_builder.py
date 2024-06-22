import pandas as pd
import networkx as nx
from itertools import combinations
from sentence_transformers import SentenceTransformer, util

def load_data(customers_csv, products_csv, purchases_csv, co_purchases_csv):
    customers_df = pd.read_csv(customers_csv)
    products_df = pd.read_csv(products_csv)
    purchases_df = pd.read_csv(purchases_csv)
    co_purchases_df = pd.read_csv(co_purchases_csv)

    purchases_df['purchase_date'] = pd.to_datetime(purchases_df['purchase_date'], errors='coerce')

    return customers_df, products_df, purchases_df, co_purchases_df

def create_graph(customers_df, products_df, purchases_df, co_purchases_df, similarity_threshold=0.7):
    G = nx.Graph()
    labels_dict = {}

    # Add customers as nodes
    for _, row in customers_df.iterrows():
        G.add_node(row['customer_id'], type='customer', name=row['customer_name'])
        labels_dict[row['customer_id']] = row['customer_name']

    # Add products as nodes
    for _, row in products_df.iterrows():
        G.add_node(row['product_id'], type='product', name=row['product_name'])
        labels_dict[row['product_id']] = row['product_name']

    # Add edges for purchases
    for _, row in purchases_df.iterrows():
        G.add_edge(row['customer_id'], row['product_id'], type='purchase', purchase_date=row['purchase_date'])

    # Add edges for co-purchases
    for _, row in co_purchases_df.iterrows():
        G.add_edge(row['product_id_1'], row['product_id_2'], type='co_purchase')

    # Add customer similarity edges
    product_to_customers = purchases_df.groupby('product_id')['customer_id'].apply(list).to_dict()
    for customers in product_to_customers.values():
        for customer1, customer2 in combinations(customers, 2):
            if G.has_edge(customer1, customer2):
                G[customer1][customer2]['weight'] += 1
            else:
                G.add_edge(customer1, customer2, type='customer_similarity', weight=1)

    # Optionally, add temporal edges (purchases made within the same month)
    purchases_df['purchase_month'] = purchases_df['purchase_date'].dt.to_period('M')
    month_to_purchases = purchases_df.groupby('purchase_month')['customer_id'].apply(list).to_dict()
    for customers in month_to_purchases.values():
        for customer1, customer2 in combinations(customers, 2):
            if G.has_edge(customer1, customer2):
                G[customer1][customer2]['weight'] += 1
            else:
                G.add_edge(customer1, customer2, type='temporal', weight=1)

    # Add semantic similarity edges between products
    model = SentenceTransformer('all-MiniLM-L6-v2')
    product_embeddings = model.encode(products_df['product_name'].tolist(), convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(product_embeddings, product_embeddings)

    for i in range(len(products_df)):
        for j in range(i + 1, len(products_df)):
            if similarities[i][j] > similarity_threshold:
                G.add_edge(products_df.iloc[i]['product_id'], products_df.iloc[j]['product_id'], type='semantic_similarity', weight=float(similarities[i][j]))

    return G, labels_dict
