import pandas as pd
import duckdb

def load_data(accounts_csv, opportunities_csv):
    accounts_df = pd.read_csv(accounts_csv)
    opportunities_df = pd.read_csv(opportunities_csv)

    accounts_df['created_date'] = pd.to_datetime(accounts_df['created_date'], errors='coerce')
    opportunities_df['created_date'] = pd.to_datetime(opportunities_df['created_date'], errors='coerce')
    opportunities_df['close_date'] = pd.to_datetime(opportunities_df['close_date'], errors='coerce')

    return accounts_df, opportunities_df

def setup_duckdb(accounts_df, opportunities_df):
    con = duckdb.connect()
    con.register('accounts_df', accounts_df)
    con.register('opportunities_df', opportunities_df)
    con.execute("CREATE OR REPLACE TABLE accounts AS SELECT * FROM accounts_df")
    con.execute("CREATE OR REPLACE TABLE opportunities AS SELECT * FROM opportunities_df")
    return con

def fetch_data(con):
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
    return result_df

def aggregate_opportunity_data(result_df):
    opportunity_agg = result_df.groupby('account_id').agg({
        'opportunity_id': 'count',
        'opportunity_amount': 'sum',
        'opportunity_probability': 'mean',
        'is_won': 'first'
    }).reset_index()
    opportunity_agg.rename(columns={
        'opportunity_id': 'opportunity_count',
        'opportunity_amount': 'total_opportunity_amount',
        'opportunity_probability': 'avg_opportunity_probability',
        'is_won': 'recent_is_won'
    }, inplace=True)

    result_df = result_df.drop_duplicates(subset=['account_id']).merge(opportunity_agg, on='account_id', how='left')
    return result_df

def get_last_opportunity_won_status(result_df):
    result_df['close_date'] = pd.to_datetime(result_df['close_date'], errors='coerce')
    sorted_df = result_df.sort_values(by=['account_id', 'close_date'])
    last_opportunity_df = sorted_df.groupby('account_id').tail(1)
    last_opportunity_won_status_df = last_opportunity_df[['account_id', 'is_won']]

    return last_opportunity_won_status_df
