from .base import Capability
from .sec_analysis import SECAnalysisCapability
from .registry import CapabilityRegistry

# Register all capabilities
CapabilityRegistry.register("sec_analysis", SECAnalysisCapability)

__all__ = [
    'Capability',
    'SECAnalysisCapability',
    'CapabilityRegistry'
] 