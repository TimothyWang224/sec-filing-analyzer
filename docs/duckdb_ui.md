# DuckDB UI

The SEC Filing Analyzer now includes integration with DuckDB UI, a powerful web interface for exploring and querying DuckDB databases.

## Features

DuckDB UI provides a rich set of features for working with DuckDB databases:

- **Database Explorer**: Browse databases, tables, and views in a hierarchical view
- **Table Summary**: View detailed information about tables, including column names, types, and data profiles
- **SQL Notebooks**: Write and execute SQL queries in a notebook interface with syntax highlighting and autocomplete
- **Column Explorer**: Analyze query results with the Column Explorer, which shows summaries and visualizations of your data
- **Data Export**: Export query results to various formats, including CSV, JSON, and Parquet

## Accessing DuckDB UI

You can access the DuckDB UI from several places in the SEC Filing Analyzer:

1. **Home Page**: Click the "Open DuckDB UI" button in the Quick Actions section
2. **Data Explorer**: Click the "Launch DuckDB UI" button in the DuckDB Tools section
3. **ETL Data Inventory**: Click the "Open DuckDB UI" button in the Data Summary section

## Using DuckDB UI

Once the DuckDB UI is launched, you can:

1. **Browse Data**: Use the left panel to browse databases, tables, and views
2. **View Table Details**: Click on a table to see its structure and a profile of its data
3. **Run Queries**: Create a new notebook or use an existing one to write and execute SQL queries
4. **Explore Results**: Use the Column Explorer on the right to analyze your query results
5. **Export Data**: Use the export controls below query results to save data in various formats

## Technical Details

- DuckDB UI is a feature of DuckDB v1.2.1 and newer
- It runs as a local web server on your machine
- Your data never leaves your computer
- The UI is accessed through your web browser
- It's implemented as a DuckDB extension called `ui`

## Troubleshooting

If you encounter issues with the DuckDB UI:

1. **DuckDB Version**: Make sure you have DuckDB v1.2.1 or newer installed
2. **Browser Issues**: If the UI doesn't open automatically, check your browser's popup blocker
3. **Connection Issues**: If you see connection errors, make sure the database file exists and is accessible
4. **Extension Issues**: If the UI extension isn't installed, DuckDB will attempt to install it automatically

For more information, visit the [DuckDB UI documentation](https://duckdb.org/docs/extensions/ui.html).
