from .registry import ToolRegistry
from .base import Tool

# Import tools - the order matters to avoid circular imports
from .sec_data import SECDataTool
from .sec_semantic_search import SECSemanticSearchTool
from .sec_graph_query import SECGraphQueryTool
from .sec_financial_data import SECFinancialDataTool
from .tool_details import ToolDetailsTool

# Register tools directly - the tool classes will be updated later to use decorators
ToolRegistry._register_tool(SECSemanticSearchTool, name="sec_semantic_search", tags=["sec", "semantic", "search"])
ToolRegistry._register_tool(SECGraphQueryTool, name="sec_graph_query", tags=["sec", "graph", "query"])
ToolRegistry._register_tool(SECFinancialDataTool, name="sec_financial_data", tags=["sec", "financial", "data"])
ToolRegistry._register_tool(SECDataTool, name="sec_data", tags=["sec", "data"])
ToolRegistry._register_tool(ToolDetailsTool, name="tool_details", tags=["meta", "tools"])

__all__ = [
    'Tool',
    'SECDataTool',
    'SECSemanticSearchTool',
    'SECGraphQueryTool',
    'SECFinancialDataTool',
    'ToolDetailsTool',
    'ToolRegistry'
]