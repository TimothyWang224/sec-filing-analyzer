"""
ETL Logs Viewer

This page displays logs and statistics for ETL runs.
"""

import time
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sec_filing_analyzer.utils.etl_logging import (
    generate_etl_report,
    get_etl_log_dir,
    get_etl_run_stats,
    get_recent_etl_runs,
)

# Page configuration
st.set_page_config(page_title="ETL Logs", page_icon="📊", layout="wide")

# Page title
st.title("ETL Logs and Statistics")

# Sidebar
st.sidebar.header("Options")
refresh_button = st.sidebar.button("Refresh Data")
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)

# Auto-refresh logic
if auto_refresh:
    refresh_interval = 30  # seconds
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()

    current_time = time.time()
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        refresh_button = True

# Main content
if refresh_button or "etl_runs" not in st.session_state:
    with st.spinner("Loading ETL runs..."):
        st.session_state.etl_runs = get_recent_etl_runs(limit=50)
        st.session_state.last_refresh = time.time()

# Display last refresh time
st.caption(f"Last refreshed: {datetime.fromtimestamp(st.session_state.last_refresh).strftime('%Y-%m-%d %H:%M:%S')}")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Recent Runs", "Run Details", "Log Explorer"])

# Tab 1: Recent Runs
with tab1:
    if not st.session_state.etl_runs:
        st.info("No ETL runs found. Run the ETL pipeline to generate logs.")
    else:
        # Convert to DataFrame for easier display
        runs_df = pd.DataFrame(st.session_state.etl_runs)

        # Format timestamps
        if "start_time" in runs_df.columns:
            runs_df["start_time"] = pd.to_datetime(runs_df["start_time"])
            runs_df["start_time_formatted"] = runs_df["start_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        if "end_time" in runs_df.columns:
            runs_df["end_time"] = pd.to_datetime(runs_df["end_time"])
            runs_df["end_time_formatted"] = runs_df["end_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
            # Calculate duration
            runs_df["duration"] = (runs_df["end_time"] - runs_df["start_time"]).dt.total_seconds()
            runs_df["duration_formatted"] = runs_df["duration"].apply(
                lambda x: f"{x:.2f}s" if pd.notnull(x) else "Running..."
            )

        # Display table of recent runs
        st.subheader("Recent ETL Runs")

        # Create a more user-friendly display table
        display_df = runs_df[["run_id", "start_time_formatted", "status", "duration_formatted"]].copy()
        display_df.columns = ["Run ID", "Start Time", "Status", "Duration"]

        # Add description if available
        if "description" in runs_df.columns:
            display_df["Description"] = runs_df["description"]

        # Color-code status
        def highlight_status(val):
            if val == "completed":
                return "background-color: #d4edda; color: #155724"
            elif val == "failed":
                return "background-color: #f8d7da; color: #721c24"
            elif val == "running":
                return "background-color: #cce5ff; color: #004085"
            return ""

        styled_df = display_df.style.applymap(highlight_status, subset=["Status"])
        st.dataframe(styled_df, use_container_width=True)

        # Create a bar chart of run durations
        if len(runs_df) > 0 and "duration" in runs_df.columns:
            completed_runs = runs_df[runs_df["status"] == "completed"].copy()
            if len(completed_runs) > 0:
                st.subheader("ETL Run Durations")

                # Create a more descriptive x-axis label
                completed_runs["run_label"] = completed_runs.apply(
                    lambda row: f"{row['run_id']} ({row['start_time_formatted']})",
                    axis=1,
                )

                # Sort by start time
                completed_runs = completed_runs.sort_values("start_time")

                # Take the last 20 runs for the chart
                chart_data = completed_runs.tail(20)

                fig = px.bar(
                    chart_data,
                    x="run_label",
                    y="duration",
                    labels={"run_label": "Run ID", "duration": "Duration (seconds)"},
                    title="ETL Run Durations (Recent Runs)",
                )

                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

# Tab 2: Run Details
with tab2:
    if not st.session_state.etl_runs:
        st.info("No ETL runs found. Run the ETL pipeline to generate logs.")
    else:
        # Create a selectbox for run selection
        run_options = [(run["run_id"], run.get("description", run["run_id"])) for run in st.session_state.etl_runs]
        selected_run_tuple = st.selectbox(
            "Select ETL Run",
            options=run_options,
            format_func=lambda x: f"{x[1]} ({x[0]})",
        )

        if selected_run_tuple:
            selected_run_id = selected_run_tuple[0]

            # Get run statistics
            run_stats = get_etl_run_stats(selected_run_id)

            if run_stats:
                # Create columns for summary stats
                col1, col2, col3, col4 = st.columns(4)

                # Calculate duration
                start_time = run_stats.get("start_time", 0)
                end_time = run_stats.get("end_time", time.time())
                duration = end_time - start_time

                with col1:
                    st.metric("Status", run_stats.get("status", "Unknown"))

                with col2:
                    st.metric("Duration", f"{duration:.2f}s")

                with col3:
                    st.metric("Filings Processed", run_stats.get("filings_processed", 0))

                with col4:
                    st.metric("Companies Processed", run_stats.get("companies_processed", 0))

                # Parameters
                st.subheader("Parameters")
                parameters = run_stats.get("parameters", {})
                if parameters:
                    params_df = pd.DataFrame([parameters])
                    st.dataframe(params_df, use_container_width=True)
                else:
                    st.info("No parameters recorded for this run.")

                # Phase Timings
                st.subheader("Phase Timings")
                phase_timings = run_stats.get("phase_timings", {})
                if phase_timings:
                    # Prepare data for chart
                    phases = []
                    durations = []

                    for phase, timings in phase_timings.items():
                        total_duration = sum(t.get("duration", 0) for t in timings)
                        phases.append(phase)
                        durations.append(total_duration)

                    # Create DataFrame
                    timing_df = pd.DataFrame({"Phase": phases, "Duration (s)": durations})

                    # Sort by duration
                    timing_df = timing_df.sort_values("Duration (s)", ascending=False)

                    # Create chart
                    fig = px.bar(timing_df, x="Phase", y="Duration (s)", title="Phase Durations")

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No phase timing data recorded for this run.")

                # Rate Limit History
                st.subheader("Rate Limit Adjustments")
                rate_limit_history = run_stats.get("rate_limit_history", [])
                if rate_limit_history:
                    # Convert to DataFrame
                    rate_df = pd.DataFrame(rate_limit_history)

                    # Convert timestamp to datetime
                    rate_df["timestamp"] = pd.to_datetime(rate_df["timestamp"])

                    # Create chart
                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=rate_df["timestamp"],
                            y=rate_df["new_rate"],
                            mode="lines+markers",
                            name="Rate Limit (s)",
                        )
                    )

                    fig.update_layout(
                        title="Rate Limit Adjustments Over Time",
                        xaxis_title="Time",
                        yaxis_title="Rate Limit (seconds)",
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Also show as table
                    st.dataframe(rate_df, use_container_width=True)
                else:
                    st.info("No rate limit adjustments recorded for this run.")

                # Errors
                st.subheader("Errors")
                errors = run_stats.get("errors", [])
                if errors:
                    # Convert to DataFrame
                    error_df = pd.DataFrame(errors)
                    st.dataframe(error_df, use_container_width=True)
                else:
                    st.success("No errors recorded for this run.")

                # Embedding Stats
                st.subheader("Embedding Statistics")
                embedding_stats = run_stats.get("embedding_stats", {})
                if embedding_stats:
                    # Convert to DataFrame
                    embed_df = pd.DataFrame([embedding_stats])
                    st.dataframe(embed_df, use_container_width=True)

                    # Create metrics
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Total Tokens", embedding_stats.get("total_tokens", 0))

                    with col2:
                        st.metric("Total Chunks", embedding_stats.get("total_chunks", 0))

                    with col3:
                        st.metric("Fallback Count", embedding_stats.get("fallback_count", 0))
                else:
                    st.info("No embedding statistics recorded for this run.")

                # Full Report
                st.subheader("Full Report")
                report = generate_etl_report(selected_run_id)
                st.text_area("ETL Run Report", report, height=300)
            else:
                st.warning(f"No statistics found for run {selected_run_id}")

# Tab 3: Log Explorer
with tab3:
    st.subheader("ETL Log Explorer")

    # Get log directory
    log_dir = get_etl_log_dir()

    # List log files
    log_files = list(log_dir.glob("*.log"))
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    if not log_files:
        st.info("No log files found.")
    else:
        # Create a selectbox for log file selection
        selected_log_file = st.selectbox(
            "Select Log File",
            options=log_files,
            format_func=lambda x: f"{x.name} ({datetime.fromtimestamp(x.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')})",
        )

        if selected_log_file:
            # Read log file
            with open(selected_log_file, "r") as f:
                log_content = f.readlines()

            # Filter options
            col1, col2 = st.columns(2)

            with col1:
                filter_level = st.multiselect(
                    "Filter by Level",
                    options=["INFO", "WARNING", "ERROR", "DEBUG"],
                    default=["INFO", "WARNING", "ERROR"],
                )

            with col2:
                filter_text = st.text_input("Filter by Text")

            # Apply filters
            filtered_logs = []
            for line in log_content:
                include_line = True

                # Apply level filter
                if filter_level:
                    level_match = False
                    for level in filter_level:
                        if f" - {level} - " in line:
                            level_match = True
                            break

                    if not level_match:
                        include_line = False

                # Apply text filter
                if filter_text and filter_text.strip() and filter_text.lower() not in line.lower():
                    include_line = False

                if include_line:
                    filtered_logs.append(line)

            # Display logs
            if filtered_logs:
                log_text = "".join(filtered_logs)
                st.text_area("Log Content", log_text, height=500)

                # Download button
                st.download_button(
                    label="Download Filtered Logs",
                    data=log_text,
                    file_name=f"filtered_{selected_log_file.name}",
                    mime="text/plain",
                )
            else:
                st.info("No logs match the selected filters.")
