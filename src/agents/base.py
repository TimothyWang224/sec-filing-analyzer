from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sec_filing_analyzer.llm import BaseLLM, OpenAILLM

@dataclass
class Goal:
    """Represents a goal for an agent to achieve."""
    name: str
    description: str

class Agent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(
        self,
        goals: List[Goal],
        capabilities: List[Any] = None,
        max_iterations: int = 10,
        max_duration_seconds: int = 180,
        llm: Optional[BaseLLM] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize an agent with its goals and capabilities.
        
        Args:
            goals: List of goals the agent aims to achieve
            capabilities: List of capabilities that extend agent behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
            llm: Optional LLM instance to use. If not provided, will create one from config
            llm_config: Optional configuration for creating an LLM instance
        """
        self.goals = goals
        self.capabilities = capabilities or []
        self.max_iterations = max_iterations
        self.max_duration_seconds = max_duration_seconds
        self.memory = []
        self.current_iteration = 0
        
        # Initialize LLM
        if llm:
            self.llm = llm
        elif llm_config:
            self.llm = OpenAILLM(**llm_config)
        else:
            # Default to GPT-4 for complex tasks
            self.llm = OpenAILLM(
                model="gpt-4-turbo-preview",
                temperature=0.7
            )
        
    @abstractmethod
    async def run(self, user_input: str, memory: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the agent with the given input.
        
        Args:
            user_input: The input to process
            memory: Optional memory to initialize with
            
        Returns:
            Dictionary containing the agent's response and any additional data
        """
        pass
    
    def add_to_memory(self, content: Dict[str, Any]):
        """Add an item to the agent's memory."""
        self.memory.append(content)
        
    def get_memory(self) -> List[Dict[str, Any]]:
        """Get the agent's current memory."""
        return self.memory
    
    def should_terminate(self) -> bool:
        """Check if the agent should terminate based on iteration count or duration."""
        return self.current_iteration >= self.max_iterations
    
    def increment_iteration(self):
        """Increment the current iteration counter."""
        self.current_iteration += 1 