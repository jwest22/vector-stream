import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

def visualize_clusters(embeddings, G, labels_dict):
    n_samples = embeddings.shape[0]
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity)
    embeddings_2d = tsne.fit_transform(embeddings)
    colors = [G.nodes[node]['recent_is_won'] for node in G.nodes()]

    plt.figure(figsize=(12, 12))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors, cmap='viridis')
    for i, node in enumerate(G.nodes()):
        plt.annotate(labels_dict[node], (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    plt.colorbar(label='Last Opp Won')
    plt.show()

def visualize_clusters_3d(embeddings, G, labels_dict):
    n_samples = embeddings.shape[0]
    perplexity = min(30, n_samples - 1)
    tsne = TSNE(n_components=3, perplexity=perplexity)
    embeddings_3d = tsne.fit_transform(embeddings)
    colors = [G.nodes[node]['recent_is_won'] for node in G.nodes()]

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=colors, cmap='viridis')

    for i, node in enumerate(G.nodes()):
        ax.text(embeddings_3d[i, 0], embeddings_3d[i, 1], embeddings_3d[i, 2], labels_dict[node])

    plt.colorbar(sc, label='Last Opp Won')
    plt.show()
