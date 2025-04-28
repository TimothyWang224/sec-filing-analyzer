"""
DuckDB Explorer

A standalone application for exploring DuckDB databases.
"""

import logging
import os
import sys

import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("duckdb_explorer.log")],
)
logger = logging.getLogger("duckdb_explorer")

# Log startup information
logger.info("DuckDB Explorer starting up")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")

# Set page config
st.set_page_config(
    page_title="DuckDB Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and description
st.title("DuckDB Explorer")
st.markdown("""
This is a standalone application for exploring DuckDB databases.
It's designed to provide a simple interface for browsing and querying DuckDB databases.
""")

# Database selection
st.sidebar.header("Database Selection")
default_db_path = "data/financial_data.duckdb"
db_path = st.sidebar.text_input("Database Path", value=default_db_path)
logger.info(f"Database path: {db_path}")

# Check if database exists
if os.path.exists(db_path):
    logger.info(f"Database file exists at {db_path}")
    st.sidebar.success(f"Database found at {db_path}")

    # Display file information
    file_size = os.path.getsize(db_path) / (1024 * 1024)  # Size in MB
    st.sidebar.write(f"Database file size: {file_size:.2f} MB")
    st.sidebar.write(f"Last modified: {os.path.getmtime(db_path)}")

    # Try to import duckdb
    try:
        logger.info("Attempting to import duckdb")
        import duckdb

        logger.info("Successfully imported duckdb")

        # Connect to database
        try:
            logger.info(f"Attempting to connect to database at {db_path}")
            conn = duckdb.connect(db_path)
            logger.info("Successfully connected to database")
            st.sidebar.success("Successfully connected to database")

            # Get list of tables
            try:
                logger.info("Executing SHOW TABLES query")
                tables = conn.execute("SHOW TABLES").fetchall()
                table_names = [table[0] for table in tables]
                logger.info(f"Found {len(table_names)} tables: {table_names}")

                if table_names:
                    # Table selection
                    selected_table = st.sidebar.selectbox("Select Table", table_names)
                    logger.info(f"Selected table: {selected_table}")

                    # Get row count
                    try:
                        logger.info(f"Executing COUNT query on {selected_table}")
                        row_count = conn.execute(
                            f"SELECT COUNT(*) FROM {selected_table}"
                        ).fetchone()[0]
                        logger.info(f"Table {selected_table} has {row_count} rows")
                        st.sidebar.write(f"Table {selected_table} has {row_count} rows")

                        # Main content
                        st.header(f"Table: {selected_table}")

                        # Tabs
                        tab1, tab2, tab3 = st.tabs(["Data", "Schema", "Query"])

                        with tab1:
                            # Sample data
                            st.subheader("Sample Data")

                            # Pagination
                            rows_per_page = st.slider(
                                "Rows per page",
                                min_value=5,
                                max_value=100,
                                value=10,
                                step=5,
                            )
                            max_pages = max(
                                1, (row_count + rows_per_page - 1) // rows_per_page
                            )
                            page = st.number_input(
                                "Page", min_value=1, max_value=max_pages, value=1
                            )

                            offset = (page - 1) * rows_per_page

                            # Get sample data
                            try:
                                logger.info(
                                    f"Fetching sample data from {selected_table}"
                                )

                                # Get schema to get column names
                                schema = conn.execute(
                                    f"DESCRIBE {selected_table}"
                                ).fetchall()
                                columns = [col[0] for col in schema]
                                logger.info(f"Table columns: {columns}")

                                # Fetch data as a list of tuples
                                sample_data_tuples = conn.execute(
                                    f"SELECT * FROM {selected_table} LIMIT {rows_per_page} OFFSET {offset}"
                                ).fetchall()
                                logger.info(
                                    f"Successfully fetched sample data: {len(sample_data_tuples)} rows"
                                )

                                # Convert to list of dictionaries for JSON display
                                sample_list = []
                                for row in sample_data_tuples:
                                    row_dict = {}
                                    for i, col in enumerate(columns):
                                        # Convert values to strings to avoid JSON serialization issues
                                        row_dict[col] = (
                                            str(row[i]) if row[i] is not None else None
                                        )
                                    sample_list.append(row_dict)

                                # Display as JSON
                                logger.info("Displaying sample data as JSON")
                                st.json(sample_list)

                                # Pagination controls
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    if st.button("Previous Page") and page > 1:
                                        page -= 1
                                        st.rerun()
                                with col3:
                                    if st.button("Next Page") and page < max_pages:
                                        page += 1
                                        st.rerun()
                                with col2:
                                    st.write(f"Page {page} of {max_pages}")
                            except Exception as e:
                                logger.error(f"Error fetching sample data: {e}")
                                st.error(f"Error fetching sample data: {e}")

                        with tab2:
                            # Schema
                            st.subheader("Table Schema")

                            try:
                                # Get schema
                                schema = conn.execute(
                                    f"DESCRIBE {selected_table}"
                                ).fetchall()

                                # Convert schema to list of dictionaries for JSON display
                                schema_list = []
                                for col in schema:
                                    schema_list.append(
                                        {
                                            "Column": col[0],
                                            "Type": col[1],
                                            "Null": col[2],
                                            "Key": col[3],
                                            "Default": str(col[4])
                                            if col[4] is not None
                                            else None,
                                            "Extra": str(col[5])
                                            if col[5] is not None
                                            else None,
                                        }
                                    )

                                # Display as JSON
                                logger.info("Displaying schema as JSON")
                                st.json(schema_list)

                                # Foreign keys (approximation)
                                st.subheader("Potential Foreign Keys")

                                fk_columns = []
                                for item in schema_list:
                                    col = item["Column"]
                                    if col.endswith("_id") or col in [
                                        "ticker",
                                        "filing_id",
                                        "fact_id",
                                    ]:
                                        fk_columns.append(col)

                                if fk_columns:
                                    # Convert foreign keys to list of dictionaries for JSON display
                                    fk_list = []
                                    for col in fk_columns:
                                        fk_list.append(
                                            {
                                                "Column": col,
                                                "Potential Referenced Table": col.replace(
                                                    "_id", ""
                                                )
                                                if col
                                                not in [
                                                    "ticker",
                                                    "filing_id",
                                                    "fact_id",
                                                ]
                                                else "companies"
                                                if col == "ticker"
                                                else "filings"
                                                if col == "filing_id"
                                                else "financial_facts",
                                            }
                                        )
                                    logger.info("Displaying foreign keys as JSON")
                                    st.json(fk_list)
                                else:
                                    st.write("No potential foreign keys found.")
                            except Exception as e:
                                logger.error(f"Error getting schema: {e}")
                                st.error(f"Error getting schema: {e}")

                        with tab3:
                            # Query
                            st.subheader("SQL Query")

                            # Sample queries
                            sample_queries = {
                                "Select All": f"SELECT * FROM {selected_table} LIMIT 100",
                                "Count Rows": f"SELECT COUNT(*) FROM {selected_table}",
                                "Group By": f"SELECT column1, COUNT(*) FROM {selected_table} GROUP BY column1 LIMIT 100",
                                "Join Example": f"SELECT a.*, b.* FROM {selected_table} a JOIN another_table b ON a.id = b.id LIMIT 100",
                            }

                            if selected_table == "companies":
                                sample_queries["Top Companies"] = (
                                    "SELECT * FROM companies ORDER BY ticker LIMIT 10"
                                )
                                sample_queries["Company Count"] = (
                                    "SELECT COUNT(*) FROM companies"
                                )

                            if selected_table == "filings":
                                sample_queries["Recent Filings"] = (
                                    "SELECT * FROM filings ORDER BY filing_date DESC LIMIT 10"
                                )
                                sample_queries["Filings by Type"] = (
                                    "SELECT filing_type, COUNT(*) FROM filings GROUP BY filing_type"
                                )
                                sample_queries["Filings by Company"] = (
                                    "SELECT ticker, COUNT(*) FROM filings GROUP BY ticker ORDER BY COUNT(*) DESC"
                                )

                            if selected_table == "financial_facts":
                                sample_queries["Recent Facts"] = (
                                    "SELECT * FROM financial_facts ORDER BY period_end_date DESC LIMIT 10"
                                )
                                sample_queries["Facts by Metric"] = (
                                    "SELECT metric_name, COUNT(*) FROM financial_facts GROUP BY metric_name ORDER BY COUNT(*) DESC"
                                )

                            if selected_table == "time_series_metrics":
                                sample_queries["Recent Metrics"] = (
                                    "SELECT * FROM time_series_metrics ORDER BY end_date DESC LIMIT 10"
                                )
                                sample_queries["Metrics by Company"] = (
                                    "SELECT ticker, COUNT(*) FROM time_series_metrics GROUP BY ticker ORDER BY COUNT(*) DESC"
                                )
                                sample_queries["Revenue Metrics"] = (
                                    "SELECT * FROM time_series_metrics WHERE metric_name = 'Revenue' ORDER BY end_date DESC LIMIT 10"
                                )

                            selected_sample = st.selectbox(
                                "Sample Queries", list(sample_queries.keys())
                            )

                            query = st.text_area(
                                "SQL Query",
                                value=sample_queries[selected_sample],
                                height=200,
                            )

                            if st.button("Run Query"):
                                try:
                                    logger.info(f"Executing query: {query}")

                                    # Get column names from the query
                                    column_names = [
                                        desc[0]
                                        for desc in conn.execute(query).description
                                    ]
                                    logger.info(f"Query columns: {column_names}")

                                    # Fetch data as a list of tuples
                                    raw_result = conn.execute(query).fetchall()
                                    logger.info(
                                        f"Successfully fetched query result: {len(raw_result)} rows"
                                    )

                                    # Convert to list of dictionaries for JSON display
                                    result_list = []
                                    for row in raw_result:
                                        row_dict = {}
                                        for i, col in enumerate(column_names):
                                            # Convert values to strings to avoid JSON serialization issues
                                            row_dict[col] = (
                                                str(row[i])
                                                if row[i] is not None
                                                else None
                                            )
                                        result_list.append(row_dict)

                                    # Display as JSON
                                    logger.info("Displaying query result as JSON")
                                    st.json(result_list)

                                    # Export options
                                    if result_list:
                                        try:
                                            # Create a DataFrame for CSV export
                                            export_df = pd.DataFrame(result_list)
                                            csv = export_df.to_csv(index=False).encode(
                                                "utf-8"
                                            )
                                            st.download_button(
                                                label="Download as CSV",
                                                data=csv,
                                                file_name="query_result.csv",
                                                mime="text/csv",
                                            )
                                        except Exception as csv_error:
                                            logger.error(
                                                f"Could not create CSV download: {csv_error}"
                                            )
                                            st.warning(
                                                f"Could not create CSV download: {csv_error}"
                                            )
                                except Exception as e:
                                    logger.error(f"Error executing query: {e}")
                                    st.error(f"Error executing query: {e}")
                    except Exception as count_error:
                        logger.error(f"Error getting row count: {count_error}")
                        st.error(f"Error getting row count: {count_error}")
                else:
                    logger.warning("No tables found in the database")
                    st.warning("No tables found in the database")
            except Exception as tables_error:
                logger.error(f"Error getting tables: {tables_error}")
                st.error(f"Error getting tables: {tables_error}")
        except Exception as conn_error:
            logger.error(f"Error connecting to database: {conn_error}")
            st.error(f"Error connecting to database: {conn_error}")
    except ImportError as import_error:
        logger.error(f"Error importing duckdb: {import_error}")
        st.error(f"Error importing duckdb: {import_error}")
else:
    logger.warning(f"Database file not found at {db_path}")
    st.warning(f"Database not found at {db_path}")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
This is a standalone DuckDB Explorer application.
It's designed to provide a simple interface for browsing and querying DuckDB databases.
""")

# Log completion
logger.info("DuckDB Explorer completed successfully")
