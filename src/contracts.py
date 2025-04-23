"""
Contracts for the Plan-Step ↔ Tool relationship.

This module defines the formal contract between planning and execution,
ensuring that tools and plan steps have a clear understanding of what
each expects and provides.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """
    Specification for a tool, defining its input schema and output format.
    
    This serves as a contract for what a tool expects and what it returns.
    """
    name: str
    input_schema: Dict[str, Any]  # parameter schema
    output_key: str  # canonical key placed in memory
    description: str = ""


class PlanStep(BaseModel):
    """
    A step in a plan, with a clear contract for what it expects and provides.
    
    The contract fields (expected_key, output_path, done_check) formalize
    the relationship between the step and the tool it uses.
    """
    step_id: int
    description: str
    tool: Optional[str] = None  # None ⇒ "thinking" step
    agent: Optional[str] = None  # Agent to execute this step, if applicable
    parameters: Dict[str, Any] = Field(default_factory=dict)

    # Contract fields
    expected_key: Optional[str] = None  # memory key that marks success
    output_path: Optional[List[str]] = None  # where in nested dict the value lands
    done_check: Optional[str] = None  # condition to evaluate for completion

    dependencies: List[int] = Field(default_factory=list)
    status: str = "pending"  # pending | completed | skipped
    completed_at: Optional[str] = None  # ISO format timestamp when completed
    skipped: Optional[bool] = None  # Whether the step was skipped


class Plan(BaseModel):
    """
    A complete plan with steps and metadata.
    """
    goal: str
    steps: List[PlanStep]
    status: str = "pending"  # pending | in_progress | completed
    created_at: Optional[str] = None  # ISO format timestamp
    completed_at: Optional[str] = None  # ISO format timestamp
    owner: str = "agent"  # Who owns this plan
    can_modify: bool = True  # Whether the plan can be modified


def extract_value(result: Dict[str, Any], path: List[str]) -> Any:
    """
    Extract a value from a nested dictionary using a path.
    
    Args:
        result: The dictionary to extract from
        path: List of keys to navigate the nested dictionary
        
    Returns:
        The extracted value, or None if the path doesn't exist
    """
    if not path:
        return result
    
    value = result
    for key in path:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return None
    
    return value
