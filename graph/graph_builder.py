import networkx as nx

def create_graph(result_df):
    G = nx.Graph()
    labels_dict = {}

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
                       recent_is_won=row['recent_is_won'])
            labels_dict[row['account_id']] = row['account_name']
    
    return G, labels_dict

def add_edges_by_industry(con, G):
    edges_query = """
    SELECT a1.id AS account1, a2.id AS account2
    FROM accounts a1
    JOIN accounts a2
    ON a1.industry = a2.industry
    AND a1.id != a2.id
    """
    industry_edges_df = con.execute(edges_query).fetchdf()
    for _, row in industry_edges_df.iterrows():
        G.add_edge(row['account1'], row['account2'])

def add_edges_by_won_status(con, G):
    edges_query = """
    WITH account_last_opp as (
        SELECT 
        a.id AS id,
        is_won as recent_is_won
    FROM accounts a
    LEFT JOIN opportunities o
    ON a.id = o.account_id
    QUALIFY row_number() OVER (PARTITION BY a.id ORDER BY o.close_date desc) = 1
    )
    SELECT 
    a1.id AS account1, 
    a2.id AS account2
    FROM account_last_opp a1
    JOIN account_last_opp a2
    ON a1.recent_is_won = a2.recent_is_won
    AND a1.id != a2.id
    """
    won_status_edges_df = con.execute(edges_query).fetchdf()
    for _, row in won_status_edges_df.iterrows():
        G.add_edge(row['account1'], row['account2'])
    """
    WITH accounts as (
        SELECT 
        a.id AS id
    FROM accounts a
    LEFT JOIN opportunities o
    ON a.id = o.account_id
    QUALIFY row_number() OVER (PARTITION BY a.id ORDER BY o.close_date desc) = 1
    )
    """