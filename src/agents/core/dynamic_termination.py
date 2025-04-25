"""
Example implementation of dynamic termination strategies for agents.
"""

import json
import logging
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Set


class DynamicTermination:
    """
    Mixin class providing dynamic termination strategies for agents.
    """

    def __init__(
        self,
        confidence_threshold: int = 85,
        similarity_threshold: float = 0.9,
        min_iterations: int = 1,
        enable_llm_assessment: bool = True,
        enable_convergence_check: bool = True,
        enable_confidence_check: bool = True,
        enable_info_gain_check: bool = True,
        **kwargs,
    ):
        """
        Initialize dynamic termination settings.

        Args:
            confidence_threshold: Minimum confidence level to terminate (0-100)
            similarity_threshold: Threshold for answer similarity (0.0-1.0)
            min_iterations: Minimum number of iterations before early termination
            enable_llm_assessment: Whether to use LLM self-assessment
            enable_convergence_check: Whether to check for answer convergence
            enable_confidence_check: Whether to check confidence levels
            enable_info_gain_check: Whether to check for information gain
        """
        self.confidence_threshold = confidence_threshold
        self.similarity_threshold = similarity_threshold
        self.min_iterations = min_iterations
        self.enable_llm_assessment = enable_llm_assessment
        self.enable_convergence_check = enable_convergence_check
        self.enable_confidence_check = enable_confidence_check
        self.enable_info_gain_check = enable_info_gain_check
        self.used_sources = set()
        self.log = logging.getLogger(self.__class__.__name__)

    def should_terminate(self) -> bool:
        """
        Check if the agent should terminate based on dynamic criteria.

        Returns:
            True if the agent should terminate, False otherwise
        """
        # Always respect max iterations
        if self.current_iteration >= self.max_iterations:
            return True

        # Always run at least min_iterations
        if self.current_iteration < self.min_iterations:
            return False

        # Check if we have any results yet
        if not self.memory:
            return False

        latest_result = self.memory[-1].get("content", {})

        # Check confidence threshold
        if self.enable_confidence_check and latest_result.get("confidence", 0) >= self.confidence_threshold:
            self.log.info(f"Terminating: Confidence threshold reached ({latest_result['confidence']}).")
            return True

        # Check for LLM termination signal
        if self.enable_llm_assessment and latest_result.get("should_terminate", False):
            self.log.info("Terminating: LLM determined answer is sufficient.")
            return True

        # Check for answer convergence if we have at least 2 iterations
        if self.enable_convergence_check and len(self.memory) >= 2:
            previous_result = self.memory[-2].get("content", {})
            if self._answers_similar(previous_result, latest_result):
                self.log.info("Terminating: Answer has converged.")
                return True

        # Check for information gain
        if self.enable_info_gain_check:
            current_sources = self._extract_sources(latest_result)
            new_sources = current_sources - self.used_sources
            if len(new_sources) == 0 and self.current_iteration > 1:
                self.log.info("Terminating: No new information sources.")
                return True
            self.used_sources.update(current_sources)

        # Continue iterations
        return False

    def _answers_similar(self, answer1: Dict[str, Any], answer2: Dict[str, Any]) -> bool:
        """
        Check if two answers are similar enough to consider converged.

        Args:
            answer1: First answer to compare
            answer2: Second answer to compare

        Returns:
            True if answers are similar, False otherwise
        """
        text1 = answer1.get("answer", "")
        text2 = answer2.get("answer", "")

        if not text1 or not text2:
            return False

        # Use SequenceMatcher for more sophisticated similarity
        similarity = SequenceMatcher(None, text1, text2).ratio()

        return similarity > self.similarity_threshold

    def _extract_sources(self, answer: Dict[str, Any]) -> Set[str]:
        """
        Extract unique information sources from an answer.

        Args:
            answer: Answer to extract sources from

        Returns:
            Set of unique source identifiers
        """
        sources = set()

        # Extract sources from semantic context
        for context in answer.get("supporting_data", {}).get("semantic_context", []):
            source_id = f"{context.get('company')}_{context.get('filing_date')}_{context.get('filing_type')}"
            sources.add(source_id)

        # Add other source types as needed

        return sources

    async def assess_answer_quality(self, question: str, answer: str) -> Dict[str, Any]:
        """
        Use the LLM to assess the quality of an answer.

        Args:
            question: The original question
            answer: The generated answer

        Returns:
            Dictionary with assessment results
        """
        assessment_prompt = f"""
        You have generated the following answer to the question "{question}":
        
        {answer}
        
        Please assess this answer on the following criteria:
        
        1. Completeness (0-100): Does the answer address all aspects of the question?
        2. Accuracy (0-100): Based on the information available, how accurate is the answer?
        3. Clarity (0-100): How clear and well-structured is the answer?
        4. Overall confidence (0-100): Overall, how confident are you in this answer?
        5. Should terminate (YES/NO): Is this answer sufficient, or would further iterations improve it significantly?
        
        Format your response as a JSON object with the following keys:
        {{
            "completeness": <score>,
            "accuracy": <score>,
            "clarity": <score>,
            "confidence": <score>,
            "should_terminate": "<YES/NO>",
            "reasoning": "<brief explanation>"
        }}
        """

        system_prompt = """You are an objective evaluator of answer quality. 
        Assess the given answer critically and honestly.
        Provide your assessment in the exact JSON format requested."""

        try:
            assessment_response = await self.llm.generate(
                prompt=assessment_prompt, system_prompt=system_prompt, temperature=0.3
            )

            # Find JSON pattern in the response
            json_match = re.search(r"({.*})", assessment_response.replace("\n", " "), re.DOTALL)
            if json_match:
                assessment_json = json.loads(json_match.group(1))
                return assessment_json
            else:
                self.log.warning("Could not extract JSON from LLM assessment response")
                return {
                    "completeness": 0,
                    "accuracy": 0,
                    "clarity": 0,
                    "confidence": 0,
                    "should_terminate": "NO",
                    "reasoning": "Failed to parse assessment",
                }

        except Exception as e:
            self.log.error(f"Error in answer assessment: {str(e)}")
            return {
                "completeness": 0,
                "accuracy": 0,
                "clarity": 0,
                "confidence": 0,
                "should_terminate": "NO",
                "reasoning": f"Error: {str(e)}",
            }
