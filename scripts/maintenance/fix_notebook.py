import json

# Load the notebook
with open("notebooks/explore_duckdb.ipynb", "r") as f:
    notebook = json.load(f)

# Add outputs field to all code cells
for cell in notebook["cells"]:
    if cell["cell_type"] == "code" and "outputs" not in cell:
        cell["outputs"] = []

# Save the notebook
with open("notebooks/explore_duckdb.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed successfully!")
