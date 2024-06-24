import matplotlib.pyplot as plt
import networkx as nx
import random
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def visualize_clusters(embeddings, G, labels_dict, subset_size=100):
    # Draw a subset of the graph for better visualization
    if len(G.nodes()) > subset_size:
        sampled_nodes = random.sample(list(G.nodes()), subset_size)
        subG = G.subgraph(sampled_nodes)
    else:
        subG = G
    
    plt.figure(figsize=(12, 8))
    
    # Use a layout algorithm for better node positioning
    pos = nx.spring_layout(subG, seed=42)
    
    clusters = nx.get_node_attributes(subG, 'cluster')
    node_colors = [clusters[node] for node in subG.nodes()]
    
    # Distinguish between customers and products using shape
    customer_nodes = [node for node, attr in subG.nodes(data=True) if attr['type'] == 'customer']
    product_nodes = [node for node, attr in subG.nodes(data=True) if attr['type'] == 'product']
    
    nx.draw_networkx_nodes(subG, pos, nodelist=customer_nodes, node_color='blue', node_size=50, label='Customers')
    nx.draw_networkx_nodes(subG, pos, nodelist=product_nodes, node_color='green', node_size=50, label='Products')
    
    nx.draw_networkx_edges(subG, pos, alpha=0.3)
    
    # Filter labels_dict to include only nodes in subG
    sub_labels_dict = {k: labels_dict[k] for k in subG.nodes() if k in labels_dict}
    nx.draw_networkx_labels(subG, pos, labels=sub_labels_dict, font_size=8)
    
    plt.legend()
    plt.title('Cluster Visualization')
    plt.show()

def visualize_interactive_graph(G, labels_dict):
    # Compute node positions using a layout algorithm
    pos = nx.spring_layout(G, seed=42)

    edge_x = []
    edge_y = []

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(labels_dict.get(node, str(node)))
        node_color.append('blue' if G.nodes[node]['type'] == 'customer' else 'green')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition='top center',
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line=dict(width=2)
        )
    )

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(edge_trace, row=1, col=1)
    fig.add_trace(node_trace, row=1, col=1)

    fig.update_layout(showlegend=False)
    fig.show()

def analyze_relationship(G, customer_node, product_node):
    if not (G.has_node(customer_node) and G.has_node(product_node)):
        return "One or both of the nodes do not exist in the graph."
    
    # Check if nodes are directly connected
    if G.has_edge(customer_node, product_node):
        distance = 1
        edge_type = G[customer_node][product_node].get('type', 'unknown')
        return f"{customer_node} and {product_node} are directly connected by an edge of type '{edge_type}'."
    
    # Calculate shortest path distance
    try:
        distance = nx.shortest_path_length(G, source=customer_node, target=product_node)
    except nx.NetworkXNoPath:
        distance = float('inf')

    if distance == float('inf'):
        return f"{customer_node} and {product_node} are not connected."
    
    # Check closeness
    if distance == 2:
        return f"{customer_node} and {product_node} are close together with a distance of {distance}, but not directly connected."
    else:
        return f"{customer_node} and {product_node} are far apart with a distance of {distance}."
