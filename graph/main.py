from data_loader import load_data, setup_duckdb, fetch_data, aggregate_opportunity_data
from graph_builder import create_graph, add_edges_by_industry, add_edges_by_won_status
from feature_encoder import get_node_features, encode_features
from node2vec_model import generate_node2vec_embeddings, load_node2vec_model, get_embeddings, perform_clustering
from visualization import visualize_clusters

def main():
    # data_loader
    accounts_df, opportunities_df = load_data('demo_data/sfdc_account_100.csv', 'demo_data/sfdc_opportunity_100.csv')
    con = setup_duckdb(accounts_df, opportunities_df)
    result_df = fetch_data(con)
    result_df = aggregate_opportunity_data(result_df)
    
    # graph_builder
    G, labels_dict = create_graph(result_df)
    add_edges_by_industry(con, G)
    add_edges_by_won_status(con, G)

    # feature_encoder
    features = get_node_features(G)
    encoded_features = encode_features(features)

    # node2vec_model
    generate_node2vec_embeddings(G, features=encoded_features)
    model = load_node2vec_model()
    embeddings = get_embeddings(model, G)
    perform_clustering(embeddings, G)

    # visualization
    visualize_clusters(embeddings, G, labels_dict)

if __name__ == "__main__":
    main()
