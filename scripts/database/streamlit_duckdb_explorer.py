"""
Streamlit DuckDB Explorer

A Streamlit app for exploring DuckDB databases.
"""

import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
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
tab1, tab2, tab3, tab4 = st.tabs(["Data", "Schema", "Query", "Visualize"])

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

with tab4:
    # Visualization
    st.subheader("Data Visualization")

    if selected_table in ["time_series_metrics", "financial_facts"]:
        # For financial data tables

        # Get available tickers
        @st.cache_data
        def get_tickers(conn):
            tickers = conn.execute("SELECT DISTINCT ticker FROM companies ORDER BY ticker").fetchall()
            return [ticker[0] for ticker in tickers]

        tickers = get_tickers(conn)

        # Get available metrics
        @st.cache_data
        def get_metrics(conn):
            metrics = conn.execute(
                "SELECT DISTINCT metric_name FROM time_series_metrics ORDER BY metric_name"
            ).fetchall()
            return [metric[0] for metric in metrics]

        metrics = get_metrics(conn)

        # Visualization options
        col1, col2, col3 = st.columns(3)

        with col1:
            selected_tickers = st.multiselect(
                "Select Companies", tickers, default=["MSFT"] if "MSFT" in tickers else tickers[:1]
            )

        with col2:
            selected_metrics = st.multiselect(
                "Select Metrics", metrics, default=["Revenue"] if "Revenue" in metrics else metrics[:1]
            )

        with col3:
            period_type = st.selectbox("Period Type", ["yearly", "quarterly"], index=0)

        if selected_tickers and selected_metrics:
            # Get data for visualization
            query = f"""
                SELECT 
                    ticker,
                    metric_name,
                    end_date,
                    value
                FROM 
                    time_series_metrics
                WHERE 
                    ticker IN ({", ".join([f"'{t}'" for t in selected_tickers])}) AND
                    metric_name IN ({", ".join([f"'{m}'" for m in selected_metrics])}) AND
                    period_type = '{period_type}'
                ORDER BY 
                    ticker, metric_name, end_date
            """

            try:
                viz_data = conn.execute(query).fetchdf()

                if not viz_data.empty:
                    # Choose visualization type
                    viz_type = st.selectbox("Visualization Type", ["Line Chart", "Bar Chart"], index=0)

                    if viz_type == "Line Chart":
                        fig = px.line(
                            viz_data,
                            x="end_date",
                            y="value",
                            color="ticker",
                            facet_row="metric_name",
                            title=f"{', '.join(selected_metrics)} for {', '.join(selected_tickers)} ({period_type})",
                            labels={"value": "Value", "end_date": "Date", "ticker": "Company"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        fig = px.bar(
                            viz_data,
                            x="end_date",
                            y="value",
                            color="ticker",
                            facet_row="metric_name",
                            barmode="group",
                            title=f"{', '.join(selected_metrics)} for {', '.join(selected_tickers)} ({period_type})",
                            labels={"value": "Value", "end_date": "Date", "ticker": "Company"},
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No data available for the selected options.")
            except Exception as e:
                st.error(f"Error generating visualization: {e}")
    else:
        # For other tables
        st.write("Select columns for visualization:")

        # Get columns
        columns = schema_df["Column"].tolist()

        col1, col2 = st.columns(2)

        with col1:
            x_column = st.selectbox("X-axis", columns, index=0)

        with col2:
            y_column = st.selectbox("Y-axis", columns, index=min(1, len(columns) - 1))

        # Get data for visualization
        try:
            query = f"""
                SELECT 
                    "{x_column}",
                    "{y_column}"
                FROM 
                    {selected_table}
                LIMIT 1000
            """

            viz_data = conn.execute(query).fetchdf()

            if not viz_data.empty:
                # Choose visualization type
                viz_type = st.selectbox("Visualization Type", ["Scatter Plot", "Bar Chart", "Line Chart"], index=0)

                if viz_type == "Scatter Plot":
                    fig = px.scatter(
                        viz_data,
                        x=x_column,
                        y=y_column,
                        title=f"{y_column} vs {x_column}",
                        labels={x_column: x_column, y_column: y_column},
                    )
                    st.plotly_chart(fig, use_container_width=True)
                elif viz_type == "Bar Chart":
                    fig = px.bar(
                        viz_data,
                        x=x_column,
                        y=y_column,
                        title=f"{y_column} by {x_column}",
                        labels={x_column: x_column, y_column: y_column},
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.line(
                        viz_data,
                        x=x_column,
                        y=y_column,
                        title=f"{y_column} over {x_column}",
                        labels={x_column: x_column, y_column: y_column},
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available for visualization.")
        except Exception as e:
            st.error(f"Error generating visualization: {e}")

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
