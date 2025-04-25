from .base import Capability
from .logging import LoggingCapability
from .multi_task_planning import MultiTaskPlanningCapability
from .planning import PlanningCapability
from .registry import CapabilityRegistry
from .sec_analysis import SECAnalysisCapability
from .time_awareness import TimeAwarenessCapability

# Register all capabilities
CapabilityRegistry.register("sec_analysis", SECAnalysisCapability)
CapabilityRegistry.register("time_awareness", TimeAwarenessCapability)
CapabilityRegistry.register("logging", LoggingCapability)
CapabilityRegistry.register("planning", PlanningCapability)
CapabilityRegistry.register("multi_task_planning", MultiTaskPlanningCapability)

__all__ = [
    "Capability",
    "SECAnalysisCapability",
    "TimeAwarenessCapability",
    "LoggingCapability",
    "PlanningCapability",
    "MultiTaskPlanningCapability",
    "CapabilityRegistry",
]
