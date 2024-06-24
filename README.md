## Data Science Projects
Welcome to my collection of data science projects. This repository houses various projects that explore different aspects of data science, from multi-modal vector embeddings to network graph analysis. Below are brief descriptions of the current projects in this repository.

### Project 1: Multi-Modal Vector Embeddings for Event Data (vector_stream)

**Description:**
This project generates multi-modal vector embeddings for event descriptions, geolocation data, and timestamps. These embeddings are available to be queried by a user via a Streamlit interface. The application dynamically categorizes each token in the user's query for appropriate embedding.

**Technologies Used:**
* Front-end: Streamlit
* Embeddings and NLP: PyTorch, Hugging Face Transformers
* Similarity Search: FAISS

### Project 2: Graph Embeddings and Clustering for Relational Data (graph_analysis)
**Description:**
This application generates a NetworkX graph object from relational data, which is used to generate embeddings for each record using Node2Vec and Sentence-BERT. It then clusters the embeddings and visualizes the results. The application includes scripts to load, preprocess, and visualize synthetic e-commerce data.

**Technologies Used:**
* Graph Construction: NetworkX
* Embeddings: Node2Vec, Sentence-BERT
* Clustering and Visualization: Various clustering algorithms and visualization tools
