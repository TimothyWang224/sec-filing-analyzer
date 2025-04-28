import asyncio
import json
import logging
from datetime import datetime

from src.environments.base import Environment
from src.errors import ParameterError, QueryTypeUnsupported
from src.tools.registry import ToolRegistry
from src.tools.validator import validate_call

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def test_tool_parameter_models():
    """Test the parameter models for all tools."""
    print("\n\n" + "=" * 80)
    print("TESTING TOOL PARAMETER MODELS")
    print("=" * 80 + "\n")

    # Get all tools
    tools = ToolRegistry.list_tools()
    print(f"Found {len(tools)} tools: {', '.join(tools.keys())}")

    # Create an environment
    environment = Environment()

    # Test each tool
    for tool_name, tool_info in tools.items():
        print(f"\nTESTING TOOL: {tool_name}")
        print("=" * 80)

        # Get the tool spec
        tool_spec = ToolRegistry.get_tool_spec(tool_name)
        if not tool_spec:
            print(f"❌ Tool spec not found for {tool_name}")
            continue

        print(f"Tool Spec: {tool_spec}")
        print(f"  Output Key: {tool_spec.output_key}")
        print(f"  Input Schema: {tool_spec.input_schema}")

        # Test each query type
        for query_type, param_model in tool_spec.input_schema.items():
            print(f"\n  Query Type: {query_type}")
            print(f"  Parameter Model: {param_model}")

            # Create test parameters
            test_params = create_test_parameters(tool_name, query_type)
            print(f"  Test Parameters: {json.dumps(test_params, indent=2)}")

            # Test validation
            try:
                validate_call(tool_name, query_type, test_params)
                print(f"  ✅ Validation passed")
            except Exception as e:
                print(f"  ❌ Validation failed: {str(e)}")
                continue

            # Test execution
            try:
                print(f"  Executing tool: {tool_name}")
                print(f"  Parameters: {json.dumps({'query_type': query_type, 'parameters': test_params}, indent=2)}")

                result = await environment.execute_action(
                    {"tool": tool_name, "args": {"query_type": query_type, "parameters": test_params}}
                )

                print(f"  Result: {json.dumps(result, indent=2) if result else 'None'}")

                # Check if the result has an output_key
                if isinstance(result, dict) and "output_key" in result:
                    print(f"  Output Key in Result: {result['output_key']}")
                    print(f"  Expected Output Key: {tool_spec.output_key}")

                    # Verify that the output_key matches the tool_spec.output_key
                    assert result["output_key"] == tool_spec.output_key, "Output key mismatch"
                    print("  ✅ Output key matches tool spec")
                else:
                    print("  ❌ Output key not found in result")
            except Exception as e:
                print(f"  ❌ Error executing tool: {str(e)}")


def create_test_parameters(tool_name, query_type):
    """Create test parameters for a tool and query type."""
    if tool_name == "sec_financial_data":
        if query_type == "financial_facts":
            return {"ticker": "AAPL", "metrics": ["Revenue"], "start_date": "2022-01-01", "end_date": "2022-12-31"}
        elif query_type == "company_info":
            return {"ticker": "AAPL"}
        elif query_type == "metrics":
            return {"ticker": "AAPL", "year": 2022}
        elif query_type == "time_series":
            return {"ticker": "AAPL", "metric": "Revenue", "start_date": "2020-01-01", "end_date": "2022-12-31"}
        elif query_type == "financial_ratios":
            return {"ticker": "AAPL", "ratios": ["PE", "PB"], "start_date": "2022-01-01", "end_date": "2022-12-31"}
        elif query_type == "custom_sql":
            return {"sql_query": "SELECT * FROM companies LIMIT 5"}
    elif tool_name == "sec_semantic_search":
        if query_type == "semantic_search":
            return {"query": "What are the risks related to supply chain?", "companies": ["AAPL"], "top_k": 5}
    elif tool_name == "sec_graph_query":
        if query_type == "company_filings":
            return {"ticker": "AAPL", "filing_types": ["10-K"], "limit": 5}
        elif query_type == "filing_sections":
            return {"accession_number": "0000320193-22-000108", "section_types": ["MD&A"], "limit": 5}
        elif query_type == "related_companies":
            return {"ticker": "AAPL", "relationship_type": "MENTIONS", "limit": 5}
        elif query_type == "filing_timeline":
            return {"ticker": "AAPL", "filing_type": "10-K", "limit": 5}
        elif query_type == "section_types":
            return {}
        elif query_type == "custom_cypher":
            return {"cypher_query": "MATCH (c:Company) RETURN c.ticker LIMIT 5"}
    elif tool_name == "sec_data":
        if query_type == "sec_data":
            return {"ticker": "AAPL", "filing_type": "10-K", "start_date": "2022-01-01", "end_date": "2022-12-31"}
    elif tool_name == "tool_details":
        if query_type == "tool_details":
            return {"tool_name": "sec_financial_data"}

    # Default empty parameters
    return {}


if __name__ == "__main__":
    asyncio.run(test_tool_parameter_models())
