"""
Task Parser

This module provides functionality for parsing multiple tasks from user input.
"""

import json
import re
import uuid

from ..llm.base import LLM
from .task_queue import Task, TaskQueue


class TaskParser:
    """Parser for identifying and extracting multiple tasks from user input."""

    def __init__(self, llm: LLM):
        """
        Initialize the task parser.

        Args:
            llm: LLM instance for task parsing
        """
        self.llm = llm

    async def parse_tasks(self, user_input: str) -> TaskQueue:
        """
        Parse multiple tasks from user input.

        Args:
            user_input: User input text

        Returns:
            TaskQueue containing the parsed tasks
        """
        # Use the LLM to identify and separate tasks
        prompt = f"""
        Identify the distinct tasks or questions in the following request:

        {user_input}

        If there are multiple tasks, separate them. If there is only one task, return it as a single item.
        For each task, provide:
        1. A clear, concise description of the task
        2. A priority level (1-5, where 5 is highest priority)
        3. Any dependencies on other tasks (if applicable)
        4. A unique identifier for the task (a short descriptive slug)

        Return the tasks as a JSON array of objects with these fields:
        - "task": The task description
        - "priority": Priority level (1-5)
        - "dependencies": Array of indices of tasks this depends on (empty if none)
        - "id": A unique identifier for the task (e.g., "apple_revenue_analysis")

        Example:
        [
          {{"task": "Analyze revenue growth for Apple", "priority": 5, "dependencies": [], "id": "apple_revenue_analysis"}},
          {{"task": "Compare profit margins with competitors", "priority": 4, "dependencies": [0], "id": "competitor_margin_comparison"}}
        ]

        If there's only one task, still return it in this format.

        IMPORTANT: Make sure each task is clearly distinct from the others. Do not create duplicate or very similar tasks.
        """

        system_prompt = """You are an expert at breaking down complex requests into clear, actionable tasks.
        Identify distinct tasks in the user's request and return them in the specified JSON format.
        If the request is simple and contains only one task, return it as a single item in the array.
        Make sure each task is specific, actionable, and focused on a single objective."""

        response = await self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.2)

        # Parse the response to extract tasks
        try:
            # Extract JSON from the response
            json_match = re.search(r"\[.*\]", response, re.DOTALL)
            if json_match:
                tasks_data = json.loads(json_match.group(0))
            else:
                # Fallback: treat the entire input as a single task
                tasks_data = [{"task": user_input, "priority": 3, "dependencies": []}]
        except Exception as e:
            print(f"Error parsing tasks: {str(e)}")
            # Fallback: treat the entire input as a single task
            tasks_data = [{"task": user_input, "priority": 3, "dependencies": []}]

        # Create task queue
        task_queue = TaskQueue()

        # Create Task objects and add them to the queue
        task_ids = []
        for i, task_data in enumerate(tasks_data):
            # Use the LLM-provided ID if available, otherwise generate one
            if "id" in task_data and task_data["id"]:
                task_id = f"task_{task_data['id']}"
            else:
                task_id = f"task_{uuid.uuid4().hex[:8]}"

            task_ids.append(task_id)

            task = Task(
                task_id=task_id,
                input_text=task_data["task"],
                priority=task_data.get("priority", 3),
            )

            # Store the original index for dependency resolution
            task.metadata["original_index"] = i

            # Store any additional metadata from the task data
            for key, value in task_data.items():
                if key not in ["task", "priority", "dependencies", "id"]:
                    task.metadata[key] = value

            task_queue.add_task(task)

        # Resolve dependencies
        for task in task_queue.get_all_tasks():
            original_index = task.metadata["original_index"]
            if original_index < len(tasks_data):
                dependency_indices = tasks_data[original_index].get("dependencies", [])
                for dep_idx in dependency_indices:
                    if 0 <= dep_idx < len(task_ids):
                        task.dependencies.append(task_ids[dep_idx])

        # Select the first task
        task_queue._select_next_task()

        return task_queue
