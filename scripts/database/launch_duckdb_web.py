"""
Launch DuckDB Web Shell

This script launches a web server that provides a browser-based interface to a DuckDB database.
"""

import argparse
import json
import os
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

import duckdb

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DuckDB Web Explorer</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.1/css/dataTables.bootstrap5.min.css">
    <style>
        body { padding: 20px; }
        .query-container { margin-bottom: 20px; }
        #results { overflow-x: auto; }
        .nav-tabs { margin-bottom: 15px; }
        .table-info { margin-bottom: 20px; }
        pre { background-color: #f8f9fa; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="container-fluid">
        <h1>DuckDB Web Explorer</h1>
        <p class="lead">Connected to: <code id="db-path"></code></p>
        
        <ul class="nav nav-tabs" id="myTab" role="tablist">
            <li class="nav-item" role="presentation">
                <button class="nav-link active" id="query-tab" data-bs-toggle="tab" data-bs-target="#query" type="button" role="tab" aria-controls="query" aria-selected="true">Query</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="tables-tab" data-bs-toggle="tab" data-bs-target="#tables" type="button" role="tab" aria-controls="tables" aria-selected="false">Tables</button>
            </li>
            <li class="nav-item" role="presentation">
                <button class="nav-link" id="schema-tab" data-bs-toggle="tab" data-bs-target="#schema" type="button" role="tab" aria-controls="schema" aria-selected="false">Schema</button>
            </li>
        </ul>
        
        <div class="tab-content" id="myTabContent">
            <!-- Query Tab -->
            <div class="tab-pane fade show active" id="query" role="tabpanel" aria-labelledby="query-tab">
                <div class="query-container">
                    <div class="mb-3">
                        <label for="sql-query" class="form-label">SQL Query:</label>
                        <textarea class="form-control" id="sql-query" rows="5" placeholder="Enter SQL query here..."></textarea>
                    </div>
                    <button id="run-query" class="btn btn-primary">Run Query</button>
                    <button id="clear-results" class="btn btn-secondary">Clear Results</button>
                </div>
                
                <div id="results">
                    <div id="query-info" class="alert alert-info d-none"></div>
                    <div id="query-error" class="alert alert-danger d-none"></div>
                    <div id="result-table" class="d-none">
                        <table id="results-table" class="table table-striped table-bordered">
                            <thead id="results-header"></thead>
                            <tbody id="results-body"></tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <!-- Tables Tab -->
            <div class="tab-pane fade" id="tables" role="tabpanel" aria-labelledby="tables-tab">
                <div class="table-info">
                    <h3>Tables</h3>
                    <div id="tables-list"></div>
                </div>
            </div>
            
            <!-- Schema Tab -->
            <div class="tab-pane fade" id="schema" role="tabpanel" aria-labelledby="schema-tab">
                <div id="schema-info">
                    <h3>Database Schema</h3>
                    <div id="schema-details"></div>
                </div>
            </div>
        </div>
    </div>

    <!-- Sample Queries Modal -->
    <div class="modal fade" id="sampleQueriesModal" tabindex="-1" aria-labelledby="sampleQueriesModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="sampleQueriesModalLabel">Sample Queries</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="list-group">
                        <button type="button" class="list-group-item list-group-item-action" data-query="SELECT * FROM companies LIMIT 10;">List companies</button>
                        <button type="button" class="list-group-item list-group-item-action" data-query="SELECT * FROM filings WHERE ticker = 'MSFT' LIMIT 10;">Microsoft filings</button>
                        <button type="button" class="list-group-item list-group-item-action" data-query="SELECT ticker, metric_name, end_date, value FROM time_series_metrics WHERE ticker = 'MSFT' AND metric_name = 'Revenue' AND period_type = 'yearly' ORDER BY end_date DESC LIMIT 10;">Microsoft yearly revenue</button>
                        <button type="button" class="list-group-item list-group-item-action" data-query="SELECT ticker, metric_name, end_date, value FROM time_series_metrics WHERE metric_name = 'Revenue' AND period_type = 'yearly' AND ticker IN ('MSFT', 'AAPL', 'GOOGL') ORDER BY ticker, end_date DESC LIMIT 30;">Compare company revenues</button>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    </div>

    <script src="https://code.jquery.com/jquery-3.6.3.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.1/js/jquery.dataTables.min.js"></script>
    <script src="https://cdn.datatables.net/1.13.1/js/dataTables.bootstrap5.min.js"></script>
    <script>
        $(document).ready(function() {
            // Set database path
            $('#db-path').text(window.location.pathname.split('/').pop());
            
            // Load tables on page load
            loadTables();
            loadSchema();
            
            // Run query button
            $('#run-query').click(function() {
                const query = $('#sql-query').val();
                if (query) {
                    runQuery(query);
                }
            });
            
            // Clear results button
            $('#clear-results').click(function() {
                $('#query-info').addClass('d-none');
                $('#query-error').addClass('d-none');
                $('#result-table').addClass('d-none');
            });
            
            // Sample queries
            $('.list-group-item').click(function() {
                const query = $(this).data('query');
                $('#sql-query').val(query);
                $('#sampleQueriesModal').modal('hide');
            });
            
            // Function to run a query
            function runQuery(query) {
                $('#query-info').addClass('d-none');
                $('#query-error').addClass('d-none');
                $('#result-table').addClass('d-none');
                
                $.ajax({
                    url: '/api/query',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({ query: query }),
                    success: function(response) {
                        if (response.error) {
                            $('#query-error').text(response.error).removeClass('d-none');
                        } else {
                            const rows = response.rows;
                            const columns = response.columns;
                            
                            // Show query info
                            $('#query-info').text(`Query executed successfully. ${rows.length} rows returned.`).removeClass('d-none');
                            
                            // Clear previous results
                            $('#results-header').empty();
                            $('#results-body').empty();
                            
                            // Create header
                            let headerRow = '<tr>';
                            columns.forEach(column => {
                                headerRow += `<th>${column}</th>`;
                            });
                            headerRow += '</tr>';
                            $('#results-header').append(headerRow);
                            
                            // Create rows
                            rows.forEach(row => {
                                let tableRow = '<tr>';
                                columns.forEach(column => {
                                    tableRow += `<td>${row[column] !== null ? row[column] : 'NULL'}</td>`;
                                });
                                tableRow += '</tr>';
                                $('#results-body').append(tableRow);
                            });
                            
                            // Show results
                            $('#result-table').removeClass('d-none');
                            
                            // Initialize DataTable
                            if ($.fn.DataTable.isDataTable('#results-table')) {
                                $('#results-table').DataTable().destroy();
                            }
                            
                            $('#results-table').DataTable({
                                paging: true,
                                searching: true,
                                ordering: true,
                                info: true,
                                pageLength: 10,
                                lengthMenu: [10, 25, 50, 100]
                            });
                        }
                    },
                    error: function(xhr, status, error) {
                        $('#query-error').text(`Error: ${error}`).removeClass('d-none');
                    }
                });
            }
            
            // Function to load tables
            function loadTables() {
                $.ajax({
                    url: '/api/tables',
                    type: 'GET',
                    success: function(response) {
                        const tables = response.tables;
                        
                        // Clear previous results
                        $('#tables-list').empty();
                        
                        // Create table list
                        let tableList = '<div class="list-group">';
                        tables.forEach(table => {
                            tableList += `<button type="button" class="list-group-item list-group-item-action table-item" data-table="${table}">${table}</button>`;
                        });
                        tableList += '</div>';
                        
                        $('#tables-list').append(tableList);
                        
                        // Add click event to table items
                        $('.table-item').click(function() {
                            const table = $(this).data('table');
                            $('#sql-query').val(`SELECT * FROM ${table} LIMIT 10;`);
                            $('#myTab button[id="query-tab"]').tab('show');
                        });
                    },
                    error: function(xhr, status, error) {
                        $('#tables-list').html(`<div class="alert alert-danger">Error loading tables: ${error}</div>`);
                    }
                });
            }
            
            // Function to load schema
            function loadSchema() {
                $.ajax({
                    url: '/api/schema',
                    type: 'GET',
                    success: function(response) {
                        const schema = response.schema;
                        
                        // Clear previous results
                        $('#schema-details').empty();
                        
                        // Create schema details
                        let schemaDetails = '';
                        Object.keys(schema).forEach(table => {
                            schemaDetails += `<h4>${table}</h4>`;
                            schemaDetails += '<table class="table table-sm table-bordered">';
                            schemaDetails += '<thead><tr><th>Column</th><th>Type</th><th>Nullable</th><th>Primary Key</th></tr></thead>';
                            schemaDetails += '<tbody>';
                            
                            schema[table].forEach(column => {
                                schemaDetails += '<tr>';
                                schemaDetails += `<td>${column.name}</td>`;
                                schemaDetails += `<td>${column.type}</td>`;
                                schemaDetails += `<td>${column.nullable ? 'Yes' : 'No'}</td>`;
                                schemaDetails += `<td>${column.primary_key ? 'Yes' : 'No'}</td>`;
                                schemaDetails += '</tr>';
                            });
                            
                            schemaDetails += '</tbody></table>';
                        });
                        
                        $('#schema-details').append(schemaDetails);
                    },
                    error: function(xhr, status, error) {
                        $('#schema-details').html(`<div class="alert alert-danger">Error loading schema: ${error}</div>`);
                    }
                });
            }
        });
    </script>
</body>
</html>
"""


class DuckDBWebHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, db_path=None, **kwargs):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())
        elif self.path == "/api/tables":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            try:
                tables = self.conn.execute("SHOW TABLES").fetchall()
                tables = [table[0] for table in tables]
                self.wfile.write(json.dumps({"tables": tables}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        elif self.path == "/api/schema":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            try:
                tables = self.conn.execute("SHOW TABLES").fetchall()
                tables = [table[0] for table in tables]

                schema = {}
                for table in tables:
                    columns = self.conn.execute(
                        f"PRAGMA table_info('{table}')"
                    ).fetchall()
                    schema[table] = []

                    for col in columns:
                        schema[table].append(
                            {
                                "name": col[1],
                                "type": col[2],
                                "nullable": not col[3],
                                "default": col[4],
                                "primary_key": col[5] == 1,
                            }
                        )

                self.wfile.write(json.dumps({"schema": schema}).encode())
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/query":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()

            try:
                query = data.get("query", "")
                result = self.conn.execute(query).fetchall()
                columns = [desc[0] for desc in self.conn.description]

                # Convert to list of dictionaries
                rows = []
                for row in result:
                    row_dict = {}
                    for i, col in enumerate(columns):
                        # Handle non-serializable types
                        if isinstance(row[i], (int, float, str, bool, type(None))):
                            row_dict[col] = row[i]
                        else:
                            row_dict[col] = str(row[i])
                    rows.append(row_dict)

                self.wfile.write(
                    json.dumps({"columns": columns, "rows": rows}).encode()
                )
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port, db_path):
    handler = lambda *args, **kwargs: DuckDBWebHandler(*args, db_path=db_path, **kwargs)
    server = HTTPServer(("localhost", port), handler)
    print(f"Starting DuckDB Web Explorer at http://localhost:{port}")
    print(f"Connected to database: {db_path}")
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Launch a web-based DuckDB explorer")
    parser.add_argument(
        "--db",
        default="data/financial_data.duckdb",
        help="Path to the DuckDB database file",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the web server on"
    )

    args = parser.parse_args()

    # Ensure the database file exists
    db_path = args.db
    if not os.path.exists(db_path):
        print(f"Database file not found: {db_path}")
        return

    # Start the server in a separate thread
    server_thread = threading.Thread(target=run_server, args=(args.port, db_path))
    server_thread.daemon = True
    server_thread.start()

    # Open the browser
    url = f"http://localhost:{args.port}"
    print(f"Opening {url} in your browser...")
    webbrowser.open(url)

    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")


if __name__ == "__main__":
    main()
