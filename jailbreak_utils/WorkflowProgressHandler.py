"""Workflow progress callback handler for LangGraph/LangChain.

This module provides a callback handler that tracks and displays the progress
of multi-turn jailbreak attack workflows in real-time with a visual flow diagram.
"""

import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set
from langchain_core.callbacks import BaseCallbackHandler


# Mapping for subgraph node names to display names
SUBGRAPH_NODE_NAME_MAP = {
    "iv_recommender_recommend_ivs": "Strategy Selection",
    "prompt_generator": "Prompt Generator",
    "iv_recommender_check_manipulation": "Manipulation Check",
}

# Display name mapping for all workflow nodes with icons
NAME_MAP = {
    "planner": "📋 Planner",
    "generator": "🔧 Generator",
    "target": "🎯 Target Attack",
    "judgement": "⚖️ Judgement",
    "iv_recommender_recommend_ivs": "Strategy Selection",
    "prompt_generator": "Prompt Generator",
    "iv_recommender_check_manipulation": "Manipulation Check",
}


class WorkflowProgressHandler(BaseCallbackHandler):
    """Callback handler for displaying real-time workflow execution progress.

    This handler tracks the execution flow through a LangGraph workflow,
    displaying elapsed time and a visual representation of the execution path
    across multiple rounds. Each round represents a complete iteration through
    the workflow nodes.

    Class Attributes:
        _round_history: History of executed steps for each round
        _start_time: Timestamp when workflow execution started
        _seen_nodes: Set of processed node keys for deduplication
        _current_langgraph_node: Currently executing node name
        _current_langgraph_step: Current step number in the workflow
        _current_round: Current round number (0-indexed)
    """

    _round_history: List[List[Dict[str, Any]]] = [[]]
    _start_time: Optional[float] = None

    _seen_nodes: Set[str] = set()

    _current_langgraph_node: Optional[str] = None
    _current_langgraph_step: Optional[int] = None

    # Current round counter for tracking multi-turn workflows
    _current_round: int = 0

    def __init__(self) -> None:
        """Initialize the progress handler and reset all tracking state."""
        super().__init__()
        WorkflowProgressHandler._round_history = [[]]
        WorkflowProgressHandler._start_time = time.time()
        WorkflowProgressHandler._seen_nodes = set()
        WorkflowProgressHandler._current_round = 0

    def _format_elapsed_time(self) -> str:
        """Format elapsed time as a human-readable string.

        Returns:
            A formatted string showing elapsed time in minutes and seconds,
            or just seconds if less than a minute has elapsed.
        """
        if self._start_time is None:
            elapsed = 0.0
        else:
            elapsed = time.time() - self._start_time

        if elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes}m {seconds:.1f}s"
        else:
            return f"{elapsed:.1f}s"

    def _add_step_to_history(self, step: int, node_name: str, iteration: int) -> None:
        """Add a step to the execution history for a specific round.

        Args:
            step: The step number in the workflow
            node_name: The name of the executed node
            iteration: The round number (0-indexed)
        """
        # Ensure the history list has enough rounds
        while iteration >= len(self._round_history):
            self._round_history.append([])

        self._round_history[iteration].append({
            "step": step,
            "name": node_name
        })

    def _format_node_name(self, node_name: str) -> str:
        """Convert internal node name to a human-readable display name.

        Args:
            node_name: The internal node identifier

        Returns:
            A formatted display name with icon, or a formatted default
            if the node name is not in the mapping.
        """
        return NAME_MAP.get(node_name, f"📦 {node_name.replace('_', ' ').title()}")

    def _refresh_display(self) -> None:
        """Refresh the terminal display with current workflow progress.

        Uses ANSI escape codes to update the display in-place without
        scrolling, showing elapsed time and execution flow for each round.
        """
        lines = []

        # Add elapsed time header
        elapsed_str = self._format_elapsed_time()
        lines.append(f"⏱️  Elapsed Time: {elapsed_str}")
        lines.append("")

        def _format_step(s: Dict[str, Any]) -> str:
            """Format a single step with optional subgraph prefix."""
            prefix = "🔄 [subgraph] " if s['name'] in SUBGRAPH_NODE_NAME_MAP else ""
            return f"{prefix}step {s['step']}: {self._format_node_name(s['name'])}"

        # Build execution flow for each round
        for _round_i, _one_round_history in enumerate(self._round_history):
            if not _one_round_history:
                continue

            steps_str = " -> ".join([
                _format_step(s)
                for s in _one_round_history
            ])
            lines.append(f"Round {_round_i} | {steps_str}")

        # Clear previous output and redraw
        sys.stdout.write("\033[u")  # Restore cursor to saved position
        sys.stdout.write("\033[J")  # Clear from cursor to end of screen

        for i, line in enumerate(lines):
            if i < len(lines) - 1:
                sys.stdout.write(line + "\n")
            else:
                sys.stdout.write(line)

        sys.stdout.flush()

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Optional[Dict[str, Any]] = None,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the start of a chain execution (LangGraph node entry).

        Tracks node execution and manages round transitions based on the
        'planner' node, which marks the beginning of each new round.

        Args:
            serialized: Serialized representation of the chain
            inputs: Inputs to the chain
            run_id: Unique identifier for this run
            parent_run_id: Unique identifier for the parent run
            tags: Tags associated with this run
            metadata: Metadata containing langgraph_node and langgraph_step
            **kwargs: Additional keyword arguments
        """
        node_name = None
        step = None

        # Extract LangGraph metadata
        if metadata and isinstance(metadata, dict):
            node_name = metadata.get("langgraph_node")
            step = metadata.get("langgraph_step")

            self._current_langgraph_node = node_name
            self._current_langgraph_step = step

        if not node_name:
            return

        # Core logic: use 'planner' node as the starting point of each round
        if node_name == "planner":
            if self._round_history[self._current_round]:
                self._current_round += 1

        iteration = self._current_round

        # Create deduplication key based on round to prevent duplicate entries
        node_key = f"{iteration}_{step}_{node_name}"

        if node_key in self._seen_nodes:
            return

        self._seen_nodes.add(node_key)

        # Add step to history (subgraph nodes are automatically attributed to current round)
        self._add_step_to_history(step, node_name, iteration)

        self._refresh_display()

    def on_chain_end(
        self,
        outputs: Optional[Dict[str, Any]] = None,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle the successful completion of a chain execution.

        Args:
            outputs: Outputs from the chain execution
            run_id: Unique identifier for this run
            parent_run_id: Unique identifier for the parent run
            **kwargs: Additional keyword arguments
        """
        pass

    def on_chain_error(
        self,
        error: Exception,
        *,
        run_id: str,
        parent_run_id: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Handle errors that occur during chain execution.

        Displays a detailed error report including the node name, step number,
        error type, error message, and a filtered traceback that excludes
        LangChain callback internals for clarity.

        Args:
            error: The exception that occurred
            run_id: Unique identifier for this run
            parent_run_id: Unique identifier for the parent run
            **kwargs: Additional keyword arguments
        """
        node_name = self._current_langgraph_node
        step = self._current_langgraph_step

        display_name = self._format_node_name(node_name) if node_name else "Unknown Node"

        error_type = type(error).__name__
        error_msg = str(error)

        # Output error header
        sys.stdout.write("\n")
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.write("❌ ERROR in Workflow Execution\n")
        sys.stdout.write("=" * 80 + "\n")
        sys.stdout.write(f"Node:     {display_name}\n")

        if node_name:
            sys.stdout.write(f"          (internal name: {node_name})\n")
        if step is not None:
            sys.stdout.write(f"Step:     {step}\n")

        sys.stdout.write(f"Run ID:   {run_id}\n\n")
        sys.stdout.write(f"Error Type:    {error_type}\n")
        sys.stdout.write(f"Error Message: {error_msg}\n")
        sys.stdout.write("-" * 80 + "\n")

        # Output filtered traceback (exclude LangChain callback internals)
        sys.stdout.write("Traceback (most recent call last):\n")

        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)

        filtered_tb = []
        for line in tb_lines:
            # Skip internal LangChain callback frames for cleaner output
            if "langchain_core" in line and "callbacks" in line:
                continue
            filtered_tb.append(line)

        sys.stdout.write("".join(filtered_tb))
        sys.stdout.write("=" * 80 + "\n\n")
        sys.stdout.flush()