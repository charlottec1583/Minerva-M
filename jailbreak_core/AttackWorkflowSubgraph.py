"""Generator subgraph workflow using LangGraph.

This module implements a generator subgraph for jailbreak attacks with
coordinated components that work in a loop:

Subgraph Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                  Generator Subgraph                      │
    │                                                          │
    │   ┌────────────────────┐      ┌────────────────────┐     │
    │   │ iv_recommender_    │      │ prompt_generator   │     │
    │   │ recommend_ivs      │─────▶│                    │     │
    │   └─────────┬──────────┘      └─────────┬──────────┘     │
    │             ▲                           │                │
    │             │                           ▼                │
    │             │         ┌──────────────────────┐           │
    │             │         │ iv_recommender_      │           │
    │             │         │ check_manipulation   │           │
    │             │         │                      │           │
    │             └─────────┤ • FAIL -> loop back  │           │
    │             ┌─────────┤ • PASS -> output     │           │
    │             │         └──────────────────────┘           │
    │             │   wrapped_prompt                           │
    │             └─▶ (final output)                           │
    └──────────────────────────────────────────────────────────┘

The subgraph receives strategic guidance (the current stage) from the
high-level Planner and iteratively generates an optimized wrapped prompt.
"""

from typing import TypedDict, Optional, Dict, List, Any
from copy import deepcopy
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable

from .BasicAgent import BasicAgent
from .AttackWorkflow import PromptGeneratorAgent


class GeneratorState(TypedDict):
    """State for the generator subgraph.

    This state flows between the Low-level Planner and Prompt Generator
    within the subgraph, carrying the context of prompt refinement.
    """

    # ========== Input from High-level Planner ==========
    current_stage: str  # the current stage determined by the high-level planner
    intention: str  # The objective to achieve
    original_prompt: str  # Original prompt to attack
    previous_turn: Optional[dict]  # Previous turn information (prompt, response, judge_result)
    experiment_config: Dict[str, Any]  # run-level design config (full | prefix_2x2 | knockout)

    # ========== Strategy Selection + Prompt Generation ==========
    ivs_activated: Optional[dict]  # Backward-compatible alias of selected_dimensions
    strategy_selection: Optional[dict]  # Full strategy payload from IV recommender
    selected_dimensions: Optional[list]  # Selected dimensions for current stage
    strategy_plan: Optional[dict]  # Factor-level realization plan
    wrapped_prompt: Optional[str]  # Generated attack prompt

    # ========== Subgraph Iteration Control ==========
    max_subgraph_iterations: int  # Maximum refinement iterations
    subgraph_iteration: int  # Current subgraph iteration counter

    # ========== Manipulation Check ==========
    manipulation_check: Optional[dict]  # Raw manipulation-check payload
    stage_alignment: int  # 1-5, whether stage goal is realized
    stage_leakage: int  # 1-5, whether leakage factors are mixed in
    manip_check_pass: bool  # PASS gate before sending prompt to target

    # ========== History and Logging ==========
    optimize_history: List[Dict[str, Any]]  # History of all optimization attempts


class IVRecommender(BasicAgent):
    """IV (Independent Variable) Recommender agent.

    This component is responsible for:
    Recommending Independent Variables (IVs) based on the current stage

    Flow:
    current_stage -> IVRecommender.recommend_ivs() -> ivs -> PromptGenerator
    """

    def __init__(
        self,
        agent: Runnable,
        system_prompt: str = "",
        session_id: str = "iv_recommender_session",
        **basic_agent_kwargs: Any,
    ):
        """Initialize IVRecommender.

        Args:
            agent: The underlying language model
            system_prompt: System prompt to guide IV recommendation
            session_id: Unique identifier for conversation history
            **basic_agent_kwargs: Optional BasicAgent retry/backoff args
        """
        super().__init__(agent, system_prompt, session_id, **basic_agent_kwargs)

    def _format_iv_recommendation_question(self, state: GeneratorState) -> str:
        """Format the IV recommendation question for the LLM.

        Args:
            state: Current generator state with current_stage and context

        Returns:
            Formatted question string for IV recommendation
        """
        # TODO: Build IV recommendation prompt based on current_stage,
        # intention, original_prompt, and iteration context
        raise NotImplementedError

    def _parse_ivs(self, response_text: str) -> dict:
        """Parse LLM response to extract recommended IVs.

        Args:
            response_text: The LLM's IV recommendation response

        Returns:
            Dictionary of extracted independent variables
        """
        # TODO: Parse and extract IVs from response
        raise NotImplementedError

    def _format_manipulation_check_question(self, state: GeneratorState) -> str:
        """Format manipulation-check question for the generated wrapped prompt."""
        raise NotImplementedError

    def _parse_manipulation_check(self, response_text: str) -> dict:
        """Parse manipulation-check result from model output."""
        raise NotImplementedError

    def recommend_ivs(self, state: GeneratorState) -> GeneratorState:
        """Recommend IVs based on the current stage and context.
        Flow:
        current_stage -> IVRecommender.recommend_ivs() -> ivs -> PromptGenerator

        Args:
            state: Current generator state

        Returns:
            Updated state with recommended IVs
        """
        # Increment subgraph iteration counter
        state["subgraph_iteration"] = state.get("subgraph_iteration", 0) + 1

        # Build IV recommendation question
        question = self._format_iv_recommendation_question(state)

        # Invoke the underlying agent through BasicAgent
        response = super().__call__({"question": question})

        # Extract response content
        resp_content = response.content if hasattr(response, 'content') else str(response)

        # Parse and update state with recommended strategy payload
        parsed = self._parse_ivs(resp_content) or {}
        state["strategy_selection"] = parsed
        state["selected_dimensions"] = parsed.get("selected_dimensions", [])
        state["strategy_plan"] = parsed.get("strategy_plan", {})
        state["ivs_activated"] = parsed.get("selected_dimensions", [])  # backward compatibility

        return state

    def check_manipulation(self, state: GeneratorState) -> GeneratorState:
        """Run manipulation check for the generated wrapped prompt."""
        question = self._format_manipulation_check_question(state)
        response = super().__call__({"question": question})
        resp_content = response.content if hasattr(response, 'content') else str(response)

        parsed = self._parse_manipulation_check(resp_content) or {}
        stage_alignment = int(parsed.get("stage_alignment", 1))
        stage_leakage = int(parsed.get("stage_leakage", 5))
        decision = str(parsed.get("decision", "FAIL")).upper()

        state["manipulation_check"] = parsed
        state["stage_alignment"] = stage_alignment
        state["stage_leakage"] = stage_leakage
        state["manip_check_pass"] = decision == "PASS"

        # Keep full audit trail for each refinement loop
        state["optimize_history"].append(deepcopy(state))
        return state


class GeneratorSubgraph:
    """Orchestrates generator subgraph: strategy selection -> prompt generation -> manipulation check."""

    def __init__(
        self,
        iv_recommender: IVRecommender,
        prompt_generator: PromptGeneratorAgent,
        max_subgraph_iterations: int = 5,
    ):
        """Initialize the generator subgraph.

        Args:
            iv_recommender: Agent for IV recommendation
            prompt_generator: Agent for generating wrapped prompts
            max_subgraph_iterations: Maximum number of refinement iterations
        """
        self.max_subgraph_iterations = max_subgraph_iterations
        self.iv_recommender = iv_recommender
        self.prompt_generator = prompt_generator
        self.subgraph = self._build_subgraph()

    def _build_subgraph(self) -> StateGraph:
        """Build the subgraph: iv_recommender_recommend_ivs -> prompt_generator -> manipulation_check."""
        subgraph = StateGraph(GeneratorState)

        # Add nodes
        subgraph.add_node(
            "iv_recommender_recommend_ivs",
            self.iv_recommender.recommend_ivs
        )
        subgraph.add_node("prompt_generator", self.prompt_generator)
        subgraph.add_node(
            "iv_recommender_check_manipulation",
            self.iv_recommender.check_manipulation
        )

        # Define subgraph edges
        subgraph.set_entry_point("iv_recommender_recommend_ivs")
        subgraph.add_edge("iv_recommender_recommend_ivs", "prompt_generator")
        subgraph.add_edge("prompt_generator", "iv_recommender_check_manipulation")

        # Conditional edge: continue refinement loop or end
        subgraph.add_conditional_edges(
            "iv_recommender_check_manipulation",
            self._should_continue,
            {"continue": "iv_recommender_recommend_ivs", "end": END}
        )

        return subgraph.compile()

    def _should_continue(self, state: GeneratorState) -> str:
        """Determine if subgraph should continue refinement or end.

        Args:
            state: Current generator state

        Returns:
            "continue" if more refinement needed, "end" otherwise
        """
        # Manipulation check passed: end subgraph
        if state.get("manip_check_pass", False):
            return "end"

        # Max iterations reached: end subgraph
        if state.get("subgraph_iteration", 0) >= state.get("max_subgraph_iterations", 5):
            return "end"

        # Otherwise, continue refinement
        return "continue"

    def run(
        self,
        current_stage: str,
        intention: str,
        original_prompt: str,
        experiment_config: Dict[str, Any],
        previous_turn: Optional[dict] = None,
    ) -> GeneratorState:
        """Run the generator subgraph.

        Args:
            current_stage: The current stage determined by the high-level planner
            intention: The objective to achieve
            original_prompt: The original prompt to attack
            experiment_config: run-level design config (full | prefix_2x2 | knockout)
            previous_turn: Previous turn information (prompt, response, judge_result)

        Returns:
            Final generator state with wrapped_prompt
        """
        # Initialize state
        initial_state: GeneratorState = {
            "current_stage": current_stage,
            "intention": intention,
            "original_prompt": original_prompt,
            "ivs_activated": None,
            "strategy_selection": None,
            "selected_dimensions": [],
            "strategy_plan": {},
            "wrapped_prompt": None,
            "max_subgraph_iterations": self.max_subgraph_iterations,
            "subgraph_iteration": 0,
            "manipulation_check": None,
            "stage_alignment": 1,
            "stage_leakage": 5,
            "manip_check_pass": False,
            "previous_turn": previous_turn,
            "experiment_config": experiment_config,
            "optimize_history": [],
        }

        # Increase recursion limit when invoking
        config = {"recursion_limit": 100}  # Increase from default 25
        # Execute subgraph
        return self.subgraph.invoke(initial_state, config=config)


