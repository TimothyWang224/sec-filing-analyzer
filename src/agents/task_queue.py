"""
Task Queue

This module provides a task queue for managing multiple tasks in agents.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


class Task:
    """Represents a single task to be processed by an agent."""

    def __init__(self, task_id: str, input_text: str, priority: int = 0):
        """
        Initialize a task.

        Args:
            task_id: Unique identifier for the task
            input_text: The input text describing the task
            priority: Priority of the task (higher number = higher priority)
        """
        self.task_id = task_id
        self.input_text = input_text
        self.priority = priority
        self.status = "pending"  # pending, in_progress, completed, failed
        self.plan = None
        self.result = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.dependencies = []  # List of task_ids that this task depends on
        self.metadata = {}  # Additional metadata about the task

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "input_text": self.input_text,
            "priority": self.priority,
            "status": self.status,
            "plan": self.plan,
            "result": self.result,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "dependencies": self.dependencies,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Create task from dictionary."""
        task = cls(
            task_id=data["task_id"],
            input_text=data["input_text"],
            priority=data.get("priority", 0),
        )
        task.status = data.get("status", "pending")
        task.plan = data.get("plan")
        task.result = data.get("result")

        # Parse datetime strings
        if data.get("created_at"):
            task.created_at = datetime.fromisoformat(data["created_at"])
        if data.get("started_at"):
            task.started_at = datetime.fromisoformat(data["started_at"])
        if data.get("completed_at"):
            task.completed_at = datetime.fromisoformat(data["completed_at"])

        task.dependencies = data.get("dependencies", [])
        task.metadata = data.get("metadata", {})
        return task


class TaskQueue:
    """A queue for managing multiple tasks."""

    def __init__(self):
        """Initialize an empty task queue."""
        self.tasks = {}  # Dict of task_id -> Task
        self.current_task_id = None
        self.task_history = []  # List of task_ids in the order they were processed

    def add_task(self, task: Task) -> str:
        """
        Add a task to the queue. If a task with the same input text already exists,
        return the existing task's ID instead of adding a duplicate.

        Args:
            task: The task to add

        Returns:
            The task_id of the added or existing task
        """
        # Check if a task with the same input already exists
        for existing_task in self.tasks.values():
            if (
                existing_task.input_text.strip().lower()
                == task.input_text.strip().lower()
            ):
                # Update priority if the new task has higher priority
                if task.priority > existing_task.priority:
                    existing_task.priority = task.priority

                # Merge dependencies
                for dep in task.dependencies:
                    if dep not in existing_task.dependencies:
                        existing_task.dependencies.append(dep)

                # Merge metadata
                for key, value in task.metadata.items():
                    existing_task.metadata[key] = value

                return existing_task.task_id

        # If no duplicate found, add the new task
        self.tasks[task.task_id] = task

        # If this is the first task, set it as the current task
        if self.current_task_id is None:
            self.current_task_id = task.task_id

        return task.task_id

    def get_current_task(self) -> Optional[Task]:
        """
        Get the current task.

        Returns:
            The current task, or None if there are no tasks
        """
        if self.current_task_id is None:
            return None
        return self.tasks.get(self.current_task_id)

    def mark_current_task_started(self) -> None:
        """Mark the current task as started."""
        if self.current_task_id is not None:
            task = self.tasks[self.current_task_id]
            task.status = "in_progress"
            task.started_at = datetime.now()

    def mark_current_task_completed(self, result: Any = None) -> None:
        """
        Mark the current task as completed and move to the next task.

        Args:
            result: The result of the task
        """
        if self.current_task_id is not None:
            task = self.tasks[self.current_task_id]

            # Only mark as completed if not already completed
            if task.status != "completed":
                task.status = "completed"
                task.completed_at = datetime.now()
                task.result = result

                # Add to history if not already there
                if self.current_task_id not in self.task_history:
                    self.task_history.append(self.current_task_id)

            # Find the next task
            self._select_next_task()

    def mark_current_task_failed(self, error: str = None) -> None:
        """
        Mark the current task as failed and move to the next task.

        Args:
            error: The error message
        """
        if self.current_task_id is not None:
            task = self.tasks[self.current_task_id]
            task.status = "failed"
            task.completed_at = datetime.now()
            if error:
                task.metadata["error"] = error

            # Add to history
            self.task_history.append(self.current_task_id)

            # Find the next task
            self._select_next_task()

    def _select_next_task(self) -> None:
        """Select the next task to process based on priority and dependencies."""
        # Reset current task
        self.current_task_id = None

        # Find all pending tasks
        pending_tasks = [
            task for task in self.tasks.values() if task.status == "pending"
        ]

        if not pending_tasks:
            return

        # Filter out tasks with unmet dependencies
        ready_tasks = []
        for task in pending_tasks:
            dependencies_met = True
            for dep_id in task.dependencies:
                if dep_id not in self.tasks or self.tasks[dep_id].status != "completed":
                    dependencies_met = False
                    break

            if dependencies_met:
                ready_tasks.append(task)

        if not ready_tasks:
            return

        # Sort by priority (highest first)
        ready_tasks.sort(key=lambda t: t.priority, reverse=True)

        # Select the highest priority task
        self.current_task_id = ready_tasks[0].task_id

    def has_pending_tasks(self) -> bool:
        """
        Check if there are any pending tasks.

        Returns:
            True if there are pending tasks, False otherwise
        """
        return any(task.status == "pending" for task in self.tasks.values())

    def has_current_task(self) -> bool:
        """
        Check if there is a current task.

        Returns:
            True if there is a current task, False otherwise
        """
        return self.current_task_id is not None

    def get_task_by_id(self, task_id: str) -> Optional[Task]:
        """
        Get a task by its ID.

        Args:
            task_id: The task ID

        Returns:
            The task, or None if not found
        """
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        """
        Get all tasks.

        Returns:
            List of all tasks
        """
        return list(self.tasks.values())

    def get_completed_tasks(self) -> List[Task]:
        """
        Get all completed tasks.

        Returns:
            List of completed tasks
        """
        return [task for task in self.tasks.values() if task.status == "completed"]

    def get_pending_tasks(self) -> List[Task]:
        """
        Get all pending tasks.

        Returns:
            List of pending tasks
        """
        return [task for task in self.tasks.values() if task.status == "pending"]

    def get_failed_tasks(self) -> List[Task]:
        """
        Get all failed tasks.

        Returns:
            List of failed tasks
        """
        return [task for task in self.tasks.values() if task.status == "failed"]

    def to_dict(self) -> Dict[str, Any]:
        """Convert task queue to dictionary."""
        return {
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "current_task_id": self.current_task_id,
            "task_history": self.task_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskQueue":
        """Create task queue from dictionary."""
        queue = cls()
        queue.current_task_id = data.get("current_task_id")
        queue.task_history = data.get("task_history", [])

        for task_id, task_data in data.get("tasks", {}).items():
            task = Task.from_dict(task_data)
            queue.tasks[task_id] = task

        return queue
