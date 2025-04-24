"""
Plan step retry utilities.

This module provides utilities for retrying plan steps with proper object conversion.
"""

import logging
from typing import Dict, Any, Union, Optional

from ...contracts import PlanStep, Plan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_to_plan_step(step_dict: Dict[str, Any]) -> PlanStep:
    """
    Convert a dictionary to a PlanStep object.
    
    Args:
        step_dict: Dictionary representation of a plan step
        
    Returns:
        PlanStep object
    """
    try:
        return PlanStep.model_validate(step_dict)
    except Exception as e:
        logger.warning(f"Error converting dictionary to PlanStep: {e}")
        # Create a minimal PlanStep with required fields
        return PlanStep(
            step_id=step_dict.get("step_id", 0),
            description=step_dict.get("description", "Unknown step"),
            tool=step_dict.get("tool"),
            parameters=step_dict.get("parameters", {})
        )


def prepare_step_for_retry(step: Union[Dict[str, Any], PlanStep]) -> PlanStep:
    """
    Prepare a step for retry by ensuring it's a PlanStep object.
    
    Args:
        step: Step to prepare (dictionary or PlanStep)
        
    Returns:
        PlanStep object ready for retry
    """
    # If it's already a PlanStep, return it
    if isinstance(step, PlanStep):
        return step
    
    # If it's a dictionary, convert it to a PlanStep
    if isinstance(step, dict):
        return convert_to_plan_step(step)
    
    # If it's something else, try to convert it to a dictionary first
    try:
        step_dict = dict(step)
        return convert_to_plan_step(step_dict)
    except Exception as e:
        logger.error(f"Error preparing step for retry: {e}")
        # Create a minimal PlanStep as a fallback
        return PlanStep(
            step_id=0,
            description="Error step",
            tool=None,
            parameters={}
        )
