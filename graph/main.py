from graph_builder import create_graph, load_data
from feature_encoder import get_node_features, encode_features
from node2vec_model import generate_node2vec_embeddings, load_node2vec_model, get_embeddings, perform_clustering
from analysis import visualize_interactive_graph

def main():
    # data_loader
    customers_df, products_df, purchases_df, co_purchases_df = load_data('demo_data/customers.csv', 'demo_data/products.csv', 'demo_data/purchases.csv', 'demo_data/co_purchases.csv')
    
    # graph_builder
    G, labels_dict = create_graph(customers_df, products_df, purchases_df, co_purchases_df)

    # feature_encoder
    features = get_node_features(G)
    encoded_features = encode_features(features)

    # node2vec_model
    generate_node2vec_embeddings(G, features=encoded_features)
    model = load_node2vec_model()
    embeddings = get_embeddings(model, G)
    perform_clustering(embeddings, G)
    
    # visualization
    visualize_interactive_graph(G, labels_dict)

if __name__ == "__main__":
    main()
