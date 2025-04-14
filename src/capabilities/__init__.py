from .base import Capability
from .sec_analysis import SECAnalysisCapability
from .time_awareness import TimeAwarenessCapability
from .logging import LoggingCapability
from .registry import CapabilityRegistry

# Register all capabilities
CapabilityRegistry.register("sec_analysis", SECAnalysisCapability)
CapabilityRegistry.register("time_awareness", TimeAwarenessCapability)
CapabilityRegistry.register("logging", LoggingCapability)

__all__ = [
    'Capability',
    'SECAnalysisCapability',
    'TimeAwarenessCapability',
    'LoggingCapability',
    'CapabilityRegistry'
]