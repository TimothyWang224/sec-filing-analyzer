"""
Simple DuckDB Explorer

A simplified Streamlit app for exploring DuckDB databases.
"""

import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

# Set page config
st.set_page_config(page_title="DuckDB Explorer", page_icon="📊", layout="wide")

# Title
st.title("DuckDB Explorer")

# Sidebar
st.sidebar.header("Configuration")

# Database selection
default_db_path = "data/financial_data.duckdb"
db_path = st.sidebar.text_input("Database Path", value=default_db_path)


# Connect to database
@st.cache_resource
def get_connection(path):
    if not os.path.exists(path):
        st.sidebar.error(f"Database file not found: {path}")
        return None
    return duckdb.connect(path)


conn = get_connection(db_path)

if conn is None:
    st.error(f"Could not connect to database: {db_path}")
    st.stop()


# Get tables
@st.cache_data
def get_tables(conn):
    tables = conn.execute("SHOW TABLES").fetchall()
    return [table[0] for table in tables]


tables = get_tables(conn)

# Sidebar - Table selection
selected_table = st.sidebar.selectbox("Select Table", tables)


# Get table schema
@st.cache_data
def get_schema(conn, table):
    schema = conn.execute(f"PRAGMA table_info('{table}')").fetchall()
    return pd.DataFrame(
        [
            {
                "Column": col[1],
                "Type": col[2],
                "Nullable": "Yes" if col[3] == 0 else "No",
                "Default": col[4],
                "Primary Key": "Yes" if col[5] == 1 else "No",
            }
            for col in schema
        ]
    )


schema_df = get_schema(conn, selected_table)


# Get row count
@st.cache_data
def get_row_count(conn, table):
    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    return count


row_count = get_row_count(conn, selected_table)

# Main content
st.header(f"Table: {selected_table}")
st.write(f"Row count: {row_count}")

# Tabs
tab1, tab2, tab3 = st.tabs(["Data", "Schema", "Query"])

with tab1:
    # Sample data
    st.subheader("Sample Data")

    # Pagination
    rows_per_page = st.slider("Rows per page", min_value=5, max_value=100, value=10, step=5)
    page = st.number_input(
        "Page", min_value=1, max_value=max(1, (row_count + rows_per_page - 1) // rows_per_page), value=1
    )

    offset = (page - 1) * rows_per_page

    # Get sample data
    @st.cache_data
    def get_sample_data(conn, table, limit, offset):
        data = conn.execute(f"SELECT * FROM {table} LIMIT {limit} OFFSET {offset}").fetchdf()
        return data

    sample_data = get_sample_data(conn, selected_table, rows_per_page, offset)
    st.dataframe(sample_data, use_container_width=True)

with tab2:
    # Schema
    st.subheader("Table Schema")
    st.dataframe(schema_df, use_container_width=True)

    # Foreign keys (approximation)
    st.subheader("Potential Foreign Keys")

    fk_columns = []
    for col in schema_df["Column"]:
        if col.endswith("_id") or col in ["ticker", "filing_id", "fact_id"]:
            fk_columns.append(col)

    if fk_columns:
        fk_df = pd.DataFrame(
            [
                {
                    "Column": col,
                    "Potential Referenced Table": col.replace("_id", "")
                    if not col in ["ticker", "filing_id", "fact_id"]
                    else "companies"
                    if col == "ticker"
                    else "filings"
                    if col == "filing_id"
                    else "financial_facts",
                }
                for col in fk_columns
            ]
        )
        st.dataframe(fk_df, use_container_width=True)
    else:
        st.write("No potential foreign keys found.")

with tab3:
    # Custom query
    st.subheader("Custom SQL Query")

    # Sample queries
    sample_queries = {
        "List all tables": "SHOW TABLES",
        f"Select all from {selected_table} (limited)": f"SELECT * FROM {selected_table} LIMIT 10",
        "Companies with most filings": """
            SELECT 
                c.ticker, 
                c.name, 
                COUNT(f.filing_id) AS filing_count
            FROM 
                companies c
            LEFT JOIN 
                filings f ON c.ticker = f.ticker
            GROUP BY 
                c.ticker, c.name
            ORDER BY 
                filing_count DESC
            LIMIT 10
        """,
        "Revenue comparison": """
            SELECT 
                ticker,
                end_date,
                value
            FROM 
                time_series_metrics
            WHERE 
                metric_name = 'Revenue' AND
                period_type = 'yearly' AND
                ticker IN ('MSFT', 'AAPL', 'GOOGL')
            ORDER BY 
                ticker, end_date
        """,
    }

    selected_sample = st.selectbox("Sample Queries", list(sample_queries.keys()))

    query = st.text_area("SQL Query", value=sample_queries[selected_sample], height=200)

    if st.button("Run Query"):
        try:
            result = conn.execute(query).fetchdf()
            st.dataframe(result, use_container_width=True)

            # Export options
            if not result.empty:
                csv = result.to_csv(index=False).encode("utf-8")
                st.download_button(label="Download as CSV", data=csv, file_name=f"query_result.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error executing query: {e}")

# Sidebar - Database info
st.sidebar.header("Database Info")
st.sidebar.write(f"Tables: {len(tables)}")


# Get total row count across all tables
@st.cache_data
def get_total_row_count(conn, tables):
    total = 0
    for table in tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        total += count
    return total


total_rows = get_total_row_count(conn, tables)
st.sidebar.write(f"Total rows: {total_rows}")

# Database size
db_size = os.path.getsize(db_path) / (1024 * 1024)  # Convert to MB
st.sidebar.write(f"Database size: {db_size:.2f} MB")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("DuckDB Explorer v1.0")
st.sidebar.markdown("Created with Streamlit")
