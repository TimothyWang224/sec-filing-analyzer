from typing import Dict, Any, List, Optional
from base import Agent, Goal
from capabilities.base import Capability
from memory.base import Memory
from sec_filing_analyzer.llm import get_agent_config

class QASpecialistAgent(Agent):
    """Agent specialized in answering financial questions and providing detailed explanations."""
    
    def __init__(
        self,
        capabilities: Optional[List[Capability]] = None,
        max_iterations: int = 10,
        max_duration_seconds: int = 180,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the QA specialist agent.
        
        Args:
            capabilities: List of capabilities to extend agent behavior
            max_iterations: Maximum number of action loops
            max_duration_seconds: Maximum runtime in seconds
            llm_config: Optional LLM configuration to override defaults
        """
        goals = [
            Goal(
                name="question_answering",
                description="Answer financial questions accurately and comprehensively"
            ),
            Goal(
                name="explanation_generation",
                description="Generate clear and detailed explanations of financial concepts"
            ),
            Goal(
                name="context_understanding",
                description="Understand and maintain context across multiple questions"
            )
        ]
        
        # Use provided config or get default from centralized config
        llm_config = llm_config or get_agent_config("qa_specialist")
        
        super().__init__(
            goals=goals,
            capabilities=capabilities,
            max_iterations=max_iterations,
            max_duration_seconds=max_duration_seconds,
            llm_config=llm_config
        )
        
    async def run(self, user_input: str, memory: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Run the QA specialist agent.
        
        Args:
            user_input: The input to process (e.g., financial question or concept)
            memory: Optional memory to initialize with
            
        Returns:
            Dictionary containing answer and supporting information
        """
        if memory:
            self.memory = memory
            
        # Initialize capabilities
        for capability in self.capabilities:
            await capability.init(self, {"input": user_input})
            
        while not self.should_terminate():
            # Start of loop capabilities
            for capability in self.capabilities:
                if not await capability.start_agent_loop(self, {"input": user_input}):
                    break
                    
            # Process the input and generate answer
            answer = await self._generate_answer(user_input)
            
            # Add result to memory
            self.add_to_memory({
                "type": "qa_response",
                "content": answer
            })
            
            # Process result with capabilities
            for capability in self.capabilities:
                answer = await capability.process_result(
                    self,
                    {"input": user_input},
                    user_input,
                    {"type": "qa_response"},
                    answer
                )
            
            self.increment_iteration()
            
        return {
            "status": "completed",
            "answer": answer,
            "memory": self.get_memory()
        }
        
    async def _generate_answer(self, input: str) -> Dict[str, Any]:
        """
        Generate answer based on input.
        
        Args:
            input: Input to process (e.g., financial question)
            
        Returns:
            Dictionary containing answer and supporting information
        """
        # This is a placeholder for the actual QA logic
        # In practice, this would use various tools to:
        # 1. Understand the question
        # 2. Retrieve relevant information
        # 3. Generate comprehensive answer
        # 4. Provide supporting evidence
        
        return {
            "input": input,
            "answer": "This is a detailed answer to the financial question...",
            "explanation": "Here's a clear explanation of the concept...",
            "supporting_data": {
                "metrics": ["Revenue", "Profit Margin", "Growth Rate"],
                "sources": ["SEC Filings", "Financial Reports"],
                "context": "Historical performance and market conditions"
            },
            "related_concepts": [
                "Financial Ratios",
                "Market Analysis",
                "Risk Assessment"
            ]
        } 