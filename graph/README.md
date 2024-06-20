# Transformer Based Graph Embedding for Node Clustering

This application generates a NetworkX graph object from relational data which is used to generate transformer embeddings for each record using Node2Vec, followed by clustering and visualization of the results. The application leverages DuckDB for efficient data processing and querying.

## Salesforce Account and Opportunity Graph Analysis Example

### Features

- Load and preprocess Salesforce account and opportunity data.
- Create a graph of accounts with edges based on shared industry and date criteria.
- Generate embeddings for accounts using Node2Vec.
- Cluster the accounts using K-Means clustering.
- Visualize the clusters using t-SNE.

### Why Use This Application for Salesforce Data?
A few key benefits from analyzing Salesforce account and opportunity data using this methodology:

  * Enhanced Relationship Insights: By visualizing the connections between accounts based on shared attributes and opportunities, sales teams can identify clusters of similar accounts, uncover hidden relationships, and better understand customer segments.
  * Improved Targeting and Strategy: Clustering accounts using advanced graph and machine learning techniques enables more precise targeting, allowing sales teams to tailor their strategies for different clusters, improving overall sales efficiency and effectiveness.
  * Data-Driven Decision Making: The use of graph-based embeddings and clustering provides a robust foundation for data-driven decision making, helping sales managers prioritize high-potential accounts and optimize resource allocation.

### Requirements

- Python 3.x
- DuckDB
- Pandas
- NetworkX
- Node2Vec
- Gensim
- Matplotlib
- Scikit-learn
- SentenceTransformers

### Installation

1. Install the required Python packages:
    ```sh
    pip install duckdb pandas networkx node2vec gensim matplotlib scikit-learn sentence-transformers
    ```

### Usage

1. Place your Salesforce account and opportunity CSV files in the `demo_data` directory, or use the already provided synthetic data. Ensure the file names match the script references. 

2. Run the script:
    ```sh
    python main.py
    ```

### Explanation of Key Steps

1. **Load and Preprocess Data:**
    - Load account and opportunity data from CSV files.
    - Convert date columns to datetime format.

2. **Data Querying with DuckDB:**
    - Create DuckDB tables for accounts and opportunities.
    - Perform SQL queries to join and filter the data.

3. **Graph Creation:**
    - Create a NetworkX graph with accounts as nodes.
    - Add edges between nodes based on shared industry and date criteria.

4. **Feature Extraction and Encoding:**
    - Extract features from the graph nodes.
    - Encode textual features using Sentence-BERT.

5. **Node2Vec Embedding:**
    - Generate node embeddings using Node2Vec.

6. **Clustering and Visualization:**
    - Cluster the embeddings using K-Means. https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans
    - Visualize the clusters using t-SNE. https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE
