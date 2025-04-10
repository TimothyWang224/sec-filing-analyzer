"""
Neo4j Graph Explorer

A utility script to explore, visualize, and analyze the structure of a Neo4j graph database.
This tool helps you understand the schema, relationships, and contents of your Neo4j database.

Usage:
    python scripts/visualization/explore_neo4j_graph.py [options]

Options:
    --summary            Show a summary of the graph structure
    --nodes LABEL        List nodes with a specific label
    --relationships TYPE List relationships of a specific type
    --schema             Show the database schema
    --sample LABEL       Show sample nodes of a specific label
    --traverse START_ID  Traverse the graph starting from a node ID
    --path FROM TO       Find paths between two node labels
"""

import argparse
import logging
import os
from typing import Dict, List, Any, Optional
import json
import tempfile
import webbrowser
from pathlib import Path

from neo4j import GraphDatabase
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

from sec_filing_analyzer.config import neo4j_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Neo4jExplorer:
    """Explorer for Neo4j graph databases."""

    def __init__(
        self,
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        """Initialize the Neo4j explorer."""
        # Get Neo4j credentials from parameters or environment variables
        self.uri = url or os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL") or neo4j_config.url
        self.user = username or os.getenv("NEO4J_USER") or os.getenv("NEO4J_USERNAME") or neo4j_config.username
        self.pwd = password or os.getenv("NEO4J_PASSWORD") or neo4j_config.password
        self.db = database or os.getenv("NEO4J_DATABASE") or neo4j_config.database

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.pwd))
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}")
            raise

    def close(self):
        """Close the Neo4j connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j connection closed")

    def get_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the graph structure."""
        summary = {
            "node_counts": self.count_nodes_by_label(),
            "relationship_counts": self.count_relationships_by_type(),
            "schema": self.get_schema()
        }
        return summary

    def count_nodes_by_label(self) -> Dict[str, int]:
        """Count nodes by label."""
        with self.driver.session(database=self.db) as session:
            # First get all labels
            labels_result = session.run("CALL db.labels() YIELD label RETURN label")
            labels = [record["label"] for record in labels_result]

            # Then count nodes for each label
            counts = {}
            for label in labels:
                count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count_record = count_result.single()
                if count_record:
                    counts[label] = count_record["count"]
                else:
                    counts[label] = 0

            return counts

    def count_relationships_by_type(self) -> Dict[str, int]:
        """Count relationships by type."""
        with self.driver.session(database=self.db) as session:
            # First get all relationship types
            types_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            rel_types = [record["relationshipType"] for record in types_result]

            # Then count relationships for each type
            counts = {}
            for rel_type in rel_types:
                count_result = session.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
                count_record = count_result.single()
                if count_record:
                    counts[rel_type] = count_record["count"]
                else:
                    counts[rel_type] = 0

            return counts

    def get_schema(self) -> Dict[str, Dict[str, Any]]:
        """Get the database schema."""
        schema = {}

        with self.driver.session(database=self.db) as session:
            # Get all node labels
            labels_result = session.run("CALL db.labels() YIELD label RETURN label")
            labels = [record["label"] for record in labels_result]

            # For each label, get a sample node and its properties
            for label in labels:
                sample_result = session.run(f"""
                    MATCH (n:{label})
                    RETURN n LIMIT 1
                """)

                sample_record = sample_result.single()
                if sample_record:
                    node = sample_record["n"]
                    properties = list(dict(node).keys())
                else:
                    properties = []

                schema[label] = {
                    "properties": properties,
                    "relationships": []
                }

            # Get all relationship types
            rel_types_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
            rel_types = [record["relationshipType"] for record in rel_types_result]

            # For each relationship type, find the connected node labels
            for rel_type in rel_types:
                rel_schema_result = session.run(f"""
                    MATCH (from)-[r:{rel_type}]->(to)
                    RETURN DISTINCT labels(from) as from_labels, labels(to) as to_labels
                    LIMIT 10
                """)

                for record in rel_schema_result:
                    from_labels = record["from_labels"]
                    to_labels = record["to_labels"]

                    for from_label in from_labels:
                        if from_label in schema:
                            schema[from_label]["relationships"].append({
                                "type": rel_type,
                                "direction": "outgoing",
                                "to_labels": to_labels
                            })

                    for to_label in to_labels:
                        if to_label in schema:
                            schema[to_label]["relationships"].append({
                                "type": rel_type,
                                "direction": "incoming",
                                "from_labels": from_labels
                            })

        return schema

    def get_nodes_by_label(self, label: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get nodes with a specific label."""
        with self.driver.session(database=self.db) as session:
            result = session.run(f"""
                MATCH (n:{label})
                RETURN n, id(n) as neo4j_id
                LIMIT $limit
            """, limit=limit)

            return [{
                "id": record["neo4j_id"],
                "properties": dict(record["n"])
            } for record in result]

    def get_relationships_by_type(self, rel_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get relationships of a specific type."""
        with self.driver.session(database=self.db) as session:
            result = session.run(f"""
                MATCH (from)-[r:{rel_type}]->(to)
                RETURN from, r, to, id(r) as rel_id
                LIMIT $limit
            """, limit=limit)

            return [{
                "id": record["rel_id"],
                "from_node": dict(record["from"]),
                "to_node": dict(record["to"]),
                "properties": dict(record["r"])
            } for record in result]

    def traverse_from_node(self, node_id: str, depth: int = 2) -> Dict[str, Any]:
        """Traverse the graph starting from a node."""
        with self.driver.session(database=self.db) as session:
            # First, find the node
            node_result = session.run("""
                MATCH (n)
                WHERE n.id = $node_id OR id(n) = $node_id_int
                RETURN n, labels(n) as labels
                LIMIT 1
            """, node_id=node_id, node_id_int=int(node_id) if node_id.isdigit() else -1)

            node_record = node_result.single()
            if not node_record:
                return {"error": f"Node with ID {node_id} not found"}

            # Then, traverse outgoing relationships
            out_result = session.run("""
                MATCH (n)-[r]->(related)
                WHERE n.id = $node_id OR id(n) = $node_id_int
                RETURN type(r) as type, related, labels(related) as related_labels
                LIMIT 100
            """, node_id=node_id, node_id_int=int(node_id) if node_id.isdigit() else -1)

            outgoing = [{
                "relationship_type": record["type"],
                "target_node": dict(record["related"]),
                "target_labels": record["related_labels"]
            } for record in out_result]

            # Traverse incoming relationships
            in_result = session.run("""
                MATCH (related)-[r]->(n)
                WHERE n.id = $node_id OR id(n) = $node_id_int
                RETURN type(r) as type, related, labels(related) as related_labels
                LIMIT 100
            """, node_id=node_id, node_id_int=int(node_id) if node_id.isdigit() else -1)

            incoming = [{
                "relationship_type": record["type"],
                "source_node": dict(record["related"]),
                "source_labels": record["related_labels"]
            } for record in in_result]

            return {
                "node": dict(node_record["n"]),
                "labels": node_record["labels"],
                "outgoing_relationships": outgoing,
                "incoming_relationships": incoming
            }

    def find_paths(self, from_label: str, to_label: str, max_depth: int = 4, limit: int = 5) -> List[Dict[str, Any]]:
        """Find paths between two node labels."""
        with self.driver.session(database=self.db) as session:
            result = session.run("""
                MATCH path = (from:{from_label})-[*1..{max_depth}]->(to:{to_label})
                RETURN path
                LIMIT $limit
            """.format(from_label=from_label, to_label=to_label, max_depth=max_depth), limit=limit)

            paths = []
            for record in result:
                path = record["path"]
                path_data = {
                    "nodes": [dict(node) for node in path.nodes],
                    "relationships": [
                        {
                            "type": rel.type,
                            "properties": dict(rel)
                        } for rel in path.relationships
                    ],
                    "length": len(path.relationships)
                }
                paths.append(path_data)

            return paths

    def visualize_graph(self, limit_nodes: int = 100, output_file: Optional[str] = None) -> str:
        """Visualize the graph structure."""
        # Create a NetworkX graph
        G = nx.DiGraph()

        with self.driver.session(database=self.db) as session:
            # Get a sample of nodes
            result = session.run("""
                MATCH (n)
                RETURN n, labels(n) as labels
                LIMIT $limit
            """, limit=limit_nodes)

            # Add nodes to the graph
            for record in result:
                node = record["n"]
                labels = record["labels"]
                node_id = node.id  # Neo4j internal ID

                # Use the first label as the node type
                node_type = labels[0] if labels else "Unknown"

                # Get a good display name for the node
                if "name" in node:
                    display_name = node["name"]
                elif "ticker" in node:
                    display_name = node["ticker"]
                elif "id" in node:
                    display_name = node["id"]
                else:
                    display_name = str(node_id)

                # Add the node to the graph
                G.add_node(node_id, label=node_type, name=display_name)

            # Get relationships between these nodes
            node_ids = [node for node in G.nodes()]
            if node_ids:
                placeholders = ", ".join([str(node_id) for node_id in node_ids])
                result = session.run(f"""
                    MATCH (n)-[r]->(m)
                    WHERE id(n) IN [{placeholders}] AND id(m) IN [{placeholders}]
                    RETURN id(n) as source, id(m) as target, type(r) as type
                """)

                # Add edges to the graph
                for record in result:
                    source = record["source"]
                    target = record["target"]
                    rel_type = record["type"]

                    G.add_edge(source, target, type=rel_type)

        # Create a visualization
        plt.figure(figsize=(12, 10))

        # Create a layout for the graph
        pos = nx.spring_layout(G, seed=42)

        # Get unique node types and relationship types
        node_types = set(nx.get_node_attributes(G, 'label').values())
        edge_types = set(nx.get_edge_attributes(G, 'type').values())

        # Create a color map for node types
        color_list = list(TABLEAU_COLORS.values())
        node_colors = {node_type: color_list[i % len(color_list)] for i, node_type in enumerate(node_types)}

        # Draw nodes by type
        for node_type in node_types:
            node_list = [node for node, data in G.nodes(data=True) if data.get('label') == node_type]
            nx.draw_networkx_nodes(
                G, pos,
                nodelist=node_list,
                node_color=node_colors[node_type],
                node_size=500,
                alpha=0.8,
                label=node_type
            )

        # Draw edges by type
        for edge_type in edge_types:
            edge_list = [(u, v) for u, v, data in G.edges(data=True) if data.get('type') == edge_type]
            nx.draw_networkx_edges(
                G, pos,
                edgelist=edge_list,
                width=1.5,
                alpha=0.7,
                edge_color='gray',
                label=edge_type
            )

        # Draw node labels
        labels = {node: data.get('name', str(node)) for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        # Add a legend
        plt.legend()

        # Remove axis
        plt.axis('off')

        # Add a title
        plt.title('Neo4j Graph Visualization')

        # Save or show the visualization
        if output_file:
            plt.savefig(output_file, bbox_inches='tight')
            return output_file
        else:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name

            plt.savefig(temp_path, bbox_inches='tight')
            plt.close()

            return temp_path

def print_json(data: Any):
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Neo4j Graph Explorer")

    # Connection options
    parser.add_argument("--url", help="Neo4j URL")
    parser.add_argument("--username", help="Neo4j username")
    parser.add_argument("--password", help="Neo4j password")
    parser.add_argument("--database", help="Neo4j database name")

    # Exploration options
    parser.add_argument("--summary", action="store_true", help="Show a summary of the graph structure")
    parser.add_argument("--nodes", help="List nodes with a specific label")
    parser.add_argument("--relationships", help="List relationships of a specific type")
    parser.add_argument("--schema", action="store_true", help="Show the database schema")
    parser.add_argument("--sample", help="Show sample nodes of a specific label")
    parser.add_argument("--traverse", help="Traverse the graph starting from a node ID")
    parser.add_argument("--path", nargs=2, metavar=("FROM", "TO"), help="Find paths between two node labels")
    parser.add_argument("--visualize", action="store_true", help="Visualize the graph structure")
    parser.add_argument("--output", help="Output file for visualization")
    parser.add_argument("--limit", type=int, default=10, help="Limit the number of results")

    args = parser.parse_args()

    try:
        # Initialize Neo4j explorer
        explorer = Neo4jExplorer(
            url=args.url,
            username=args.username,
            password=args.password,
            database=args.database
        )

        # Execute the requested command
        if args.summary:
            summary = explorer.get_graph_summary()
            print("\n=== Neo4j Graph Summary ===")
            print("\nNode Counts:")
            for label, count in summary["node_counts"].items():
                print(f"  {label}: {count}")

            print("\nRelationship Counts:")
            for rel_type, count in summary["relationship_counts"].items():
                print(f"  {rel_type}: {count}")

            print("\nSchema Overview:")
            for node_type, schema_info in summary["schema"].items():
                print(f"\n  {node_type}:")
                print(f"    Properties: {', '.join(schema_info['properties'])}")
                print(f"    Relationships:")
                for rel in schema_info["relationships"]:
                    if rel["direction"] == "outgoing":
                        print(f"      -[{rel['type']}]-> {rel['to_labels']}")
                    else:
                        print(f"      <-[{rel['type']}]- {rel['from_labels']}")

        elif args.nodes:
            nodes = explorer.get_nodes_by_label(args.nodes, limit=args.limit)
            print(f"\n=== Nodes with Label '{args.nodes}' ===")
            print_json(nodes)

        elif args.relationships:
            rels = explorer.get_relationships_by_type(args.relationships, limit=args.limit)
            print(f"\n=== Relationships of Type '{args.relationships}' ===")
            print_json(rels)

        elif args.schema:
            schema = explorer.get_schema()
            print("\n=== Neo4j Database Schema ===")
            print_json(schema)

        elif args.sample:
            nodes = explorer.get_nodes_by_label(args.sample, limit=args.limit)
            print(f"\n=== Sample Nodes with Label '{args.sample}' ===")
            print_json(nodes)

        elif args.traverse:
            traversal = explorer.traverse_from_node(args.traverse)
            print(f"\n=== Traversal from Node '{args.traverse}' ===")
            print_json(traversal)

        elif args.path:
            from_label, to_label = args.path
            paths = explorer.find_paths(from_label, to_label, limit=args.limit)
            print(f"\n=== Paths from '{from_label}' to '{to_label}' ===")
            print_json(paths)

        elif args.visualize:
            output_file = args.output
            print("\n=== Visualizing Neo4j Graph ===")
            print("This may take a moment...")

            # Generate the visualization
            viz_file = explorer.visualize_graph(limit_nodes=args.limit, output_file=output_file)

            # Open the visualization in the default browser
            print(f"\nVisualization saved to: {viz_file}")
            webbrowser.open(f"file://{os.path.abspath(viz_file)}")

        else:
            # If no specific command is given, show a menu
            print("\n=== Neo4j Graph Explorer ===")
            print("\nAvailable commands:")
            print("  1. Show graph summary")
            print("  2. List nodes by label")
            print("  3. List relationships by type")
            print("  4. Show database schema")
            print("  5. Traverse from a node")
            print("  6. Find paths between node types")
            print("  7. Visualize graph structure")
            print("  0. Exit")

            choice = input("\nEnter your choice (0-7): ")

            if choice == "1":
                summary = explorer.get_graph_summary()
                print("\n=== Neo4j Graph Summary ===")
                print_json(summary)
            elif choice == "2":
                label = input("Enter node label: ")
                limit = input("Enter limit (default: 10): ")
                limit = int(limit) if limit.isdigit() else 10
                nodes = explorer.get_nodes_by_label(label, limit=limit)
                print(f"\n=== Nodes with Label '{label}' ===")
                print_json(nodes)
            elif choice == "3":
                rel_type = input("Enter relationship type: ")
                limit = input("Enter limit (default: 10): ")
                limit = int(limit) if limit.isdigit() else 10
                rels = explorer.get_relationships_by_type(rel_type, limit=limit)
                print(f"\n=== Relationships of Type '{rel_type}' ===")
                print_json(rels)
            elif choice == "4":
                schema = explorer.get_schema()
                print("\n=== Neo4j Database Schema ===")
                print_json(schema)
            elif choice == "5":
                node_id = input("Enter node ID: ")
                traversal = explorer.traverse_from_node(node_id)
                print(f"\n=== Traversal from Node '{node_id}' ===")
                print_json(traversal)
            elif choice == "6":
                from_label = input("Enter source node label: ")
                to_label = input("Enter target node label: ")
                limit = input("Enter limit (default: 5): ")
                limit = int(limit) if limit.isdigit() else 5
                paths = explorer.find_paths(from_label, to_label, limit=limit)
                print(f"\n=== Paths from '{from_label}' to '{to_label}' ===")
                print_json(paths)
            elif choice == "7":
                limit = input("Enter node limit (default: 50): ")
                limit = int(limit) if limit.isdigit() else 50
                output = input("Enter output file path (leave empty for temporary file): ")
                output_file = output if output else None

                print("\n=== Visualizing Neo4j Graph ===")
                print("This may take a moment...")

                # Generate the visualization
                viz_file = explorer.visualize_graph(limit_nodes=limit, output_file=output_file)

                # Open the visualization in the default browser
                print(f"\nVisualization saved to: {viz_file}")
                webbrowser.open(f"file://{os.path.abspath(viz_file)}")
            elif choice == "0":
                print("Goodbye!")
            else:
                print("Invalid choice")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
    finally:
        if 'explorer' in locals():
            explorer.close()

if __name__ == "__main__":
    main()
