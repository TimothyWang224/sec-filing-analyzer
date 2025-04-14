"""
Workflow Base Module

This module provides the base class for workflows that orchestrate multiple agents.
"""

import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Type, Union

from ..agents.base import Agent
from ..sec_filing_analyzer.utils.logging_utils import get_standard_log_dir

class WorkflowLogger:
    """Logger for entire workflows involving multiple agents."""
    
    def __init__(
        self,
        workflow_id: str,
        log_level: str = "INFO",
        include_agent_details: bool = True
    ):
        """
        Initialize the workflow logger.
        
        Args:
            workflow_id: Unique identifier for the workflow
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            include_agent_details: Whether to include detailed agent logs
        """
        self.workflow_id = workflow_id
        self.log_level = self._get_log_level(log_level)
        self.include_agent_details = include_agent_details
        
        # Set up logging
        self._setup_logger()
        
        # Track agents in this workflow
        self.agents = set()
    
    def _get_log_level(self, log_level: str) -> int:
        """Convert string log level to logging constant."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return level_map.get(log_level.upper(), logging.INFO)
        
    def _setup_logger(self):
        """Set up the workflow logger."""
        # Create log directory
        log_dir = Path(get_standard_log_dir("workflows"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"workflow.{self.workflow_id}")
        self.logger.setLevel(self.log_level)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler
        self.log_file = log_dir / f"workflow_{self.workflow_id}.log"
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(self.log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Create JSON log file
        self.json_log_file = log_dir / f"workflow_{self.workflow_id}.json"
        with open(self.json_log_file, 'w') as f:
            json.dump({
                "workflow_id": self.workflow_id,
                "start_time": datetime.now().isoformat(),
                "logs": []
            }, f, indent=2)
    
    def register_agent(self, agent_name: str, agent: Agent):
        """Register an agent with this workflow."""
        self.agents.add(agent)
        self.info(f"Registered agent: {agent_name} ({agent.__class__.__name__})")
        
        # Create a log handler that forwards to workflow logger
        class WorkflowLogHandler(logging.Handler):
            def __init__(self, workflow_logger, agent_name):
                super().__init__()
                self.workflow_logger = workflow_logger
                self.agent_name = agent_name
                
            def emit(self, record):
                # Create a copy of the record for the workflow logger
                workflow_record = logging.LogRecord(
                    name=record.name,
                    level=record.levelno,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg=f"[{self.agent_name}] {record.getMessage()}",
                    args=(),
                    exc_info=record.exc_info
                )
                self.workflow_logger.logger.handle(workflow_record)
        
        # Add the workflow handler to the agent's logger
        workflow_handler = WorkflowLogHandler(self, agent_name)
        workflow_handler.setLevel(self.log_level)
        agent.logger.addHandler(workflow_handler)
    
    def debug(self, message: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(message, *args, **kwargs)
        self._append_to_json_log("DEBUG", message)
    
    def info(self, message: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(message, *args, **kwargs)
        self._append_to_json_log("INFO", message)
    
    def warning(self, message: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(message, *args, **kwargs)
        self._append_to_json_log("WARNING", message)
    
    def error(self, message: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(message, *args, **kwargs)
        self._append_to_json_log("ERROR", message)
    
    def critical(self, message: str, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(message, *args, **kwargs)
        self._append_to_json_log("CRITICAL", message)
    
    def _append_to_json_log(self, level: str, message: str):
        """Append a log entry to the JSON log file."""
        try:
            with open(self.json_log_file, 'r+') as f:
                data = json.load(f)
                data["logs"].append({
                    "timestamp": datetime.now().isoformat(),
                    "level": level,
                    "message": message
                })
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
        except (json.JSONDecodeError, FileNotFoundError):
            # If there's an error, recreate the file
            with open(self.json_log_file, 'w') as f:
                json.dump({
                    "workflow_id": self.workflow_id,
                    "start_time": datetime.now().isoformat(),
                    "logs": [{
                        "timestamp": datetime.now().isoformat(),
                        "level": level,
                        "message": message
                    }]
                }, f, indent=2)
    
    def log_workflow_start(self, description: Optional[str] = None):
        """Log the start of a workflow."""
        self.info(f"Workflow started: {description or 'No description'}")
    
    def log_workflow_end(self, status: str = "completed", details: Optional[str] = None):
        """Log the end of a workflow."""
        self.info(f"Workflow {status}: {details or 'No details'}")
        
        # Add end time to JSON log
        try:
            with open(self.json_log_file, 'r+') as f:
                data = json.load(f)
                data["end_time"] = datetime.now().isoformat()
                data["status"] = status
                if details:
                    data["details"] = details
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=2)
        except (json.JSONDecodeError, FileNotFoundError):
            pass


class Workflow:
    """Base class for workflows that orchestrate multiple agents."""
    
    def __init__(
        self,
        workflow_id: Optional[str] = None,
        log_level: str = "INFO",
        description: Optional[str] = None
    ):
        """
        Initialize a workflow.
        
        Args:
            workflow_id: Unique identifier for the workflow (default: auto-generated)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            description: Description of the workflow
        """
        # Generate workflow ID if not provided
        self.workflow_id = workflow_id or f"{self.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.description = description or self.__class__.__name__
        
        # Initialize logger
        self.logger = WorkflowLogger(
            workflow_id=self.workflow_id,
            log_level=log_level
        )
        
        # Initialize agents dict
        self.agents = {}
        
        # Log workflow initialization
        self.logger.log_workflow_start(self.description)
    
    def add_agent(self, name: str, agent: Agent) -> Agent:
        """
        Add an agent to the workflow.
        
        Args:
            name: Name to identify the agent in the workflow
            agent: The agent instance to add
            
        Returns:
            The added agent
        """
        self.agents[name] = agent
        self.logger.register_agent(name, agent)
        return agent
    
    async def run(self, *args, **kwargs):
        """
        Run the workflow. To be implemented by subclasses.
        
        Returns:
            Workflow results
        """
        raise NotImplementedError("Subclasses must implement run()")
    
    def log_step(self, step_name: str, details: Optional[str] = None):
        """
        Log a workflow step.
        
        Args:
            step_name: Name of the step
            details: Optional details about the step
        """
        self.logger.info(f"Step: {step_name} - {details or ''}")
    
    def log_agent_call(self, agent_name: str, input_data: Any):
        """
        Log an agent being called.
        
        Args:
            agent_name: Name of the agent being called
            input_data: Input data being passed to the agent
        """
        input_str = str(input_data)
        if len(input_str) > 100:
            input_str = input_str[:97] + "..."
        self.logger.info(f"Calling agent: {agent_name} with input: {input_str}")
    
    def log_agent_result(self, agent_name: str, result: Any):
        """
        Log the result from an agent.
        
        Args:
            agent_name: Name of the agent that produced the result
            result: The result from the agent
        """
        result_str = str(result)
        if len(result_str) > 100:
            result_str = result_str[:97] + "..."
        self.logger.info(f"Result from {agent_name}: {result_str}")
