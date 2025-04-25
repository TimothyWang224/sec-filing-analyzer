from dataclasses import dataclass


@dataclass
class Goal:
    """Represents a goal for an agent to achieve."""

    name: str
    description: str
