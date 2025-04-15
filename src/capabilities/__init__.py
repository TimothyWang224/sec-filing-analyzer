from .base import Capability
from .sec_analysis import SECAnalysisCapability
from .time_awareness import TimeAwarenessCapability
from .logging import LoggingCapability
from .planning import PlanningCapability
from .multi_task_planning import MultiTaskPlanningCapability
from .registry import CapabilityRegistry

# Register all capabilities
CapabilityRegistry.register("sec_analysis", SECAnalysisCapability)
CapabilityRegistry.register("time_awareness", TimeAwarenessCapability)
CapabilityRegistry.register("logging", LoggingCapability)
CapabilityRegistry.register("planning", PlanningCapability)
CapabilityRegistry.register("multi_task_planning", MultiTaskPlanningCapability)

__all__ = [
    'Capability',
    'SECAnalysisCapability',
    'TimeAwarenessCapability',
    'LoggingCapability',
    'PlanningCapability',
    'MultiTaskPlanningCapability',
    'CapabilityRegistry'
]