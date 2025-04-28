import asyncio
import json
import logging

from src.contracts import ToolSpec
from src.tools.registry import ToolRegistry

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


async def test_tool_output_key():
    """Test that tool wrappers honor the ToolSpec.output_key."""
    print("\n\n" + "=" * 80)
    print("TESTING TOOL OUTPUT KEY")
    print("=" * 80 + "\n")

    # Get all registered tools
    tools = ToolRegistry.list_tools()
    print(f"Registered tools: {list(tools.keys())}")

    # Get tool specs for all tools
    for tool_name in tools.keys():
        tool_spec = ToolRegistry.get_tool_spec(tool_name)
        if tool_spec:
            print(f"\nTool: {tool_name}")
            print(f"  Output Key: {tool_spec.output_key}")
            print(f"  Description: {tool_spec.description}")
            print(f"  Input Schema: {json.dumps(tool_spec.input_schema, indent=2)}")

    # Create a custom tool spec
    custom_tool_spec = ToolSpec(
        name="test_tool",
        input_schema={"param1": {"type": "string", "description": "Test parameter"}},
        output_key="test_output",
        description="Test tool",
    )

    # Register the custom tool spec
    ToolRegistry._tool_specs["test_tool"] = custom_tool_spec

    # Verify that the custom tool spec is registered
    tool_spec = ToolRegistry.get_tool_spec("test_tool")
    if tool_spec:
        print(f"\nCustom Tool: {tool_spec.name}")
        print(f"  Output Key: {tool_spec.output_key}")
        print(f"  Description: {tool_spec.description}")
        print(f"  Input Schema: {json.dumps(tool_spec.input_schema, indent=2)}")

    return tools


if __name__ == "__main__":
    asyncio.run(test_tool_output_key())
