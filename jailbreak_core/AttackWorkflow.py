"""Multi-turn attack workflow using LangGraph.

This module implements a workflow orchestration system for multi-turn
jailbreak attacks with four coordinated agents:

Workflow Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                    Attack Workflow                       │
    │                                                          │
    │  ┌─────────┐    ┌────────────┐    ┌──────────┐           │
    │  │ Planner │ ──▶│ Generator  │ ──▶│  Target  │           │
    │  │  Agent  │    │ Subgraph   │    │  Agent   │           │
    │  └─────────┘    └────────────┘    └────┬─────┘           │
    │       ▲                                │                 │
    │       │                                ▼                 │
    │       │                           ┌──────────┐           │
    │       │                           │ Judgement│           │
    │       │                           │  Agent   │           │
    │       │                           └────┬─────┘           │
    │       │                                │                 │
    │       │            ┌───────────────────┴───────┐         │
    │       │            │  _should_continue()       │         │
    │       │            │  - success? -> END        │         │
    │       │            │  - max_iter? -> END       │         │
    │       │            │  - else -> Planner        │         │
    │       └────────────┴───────────────────────────┘         │
    │                                                          │
    └──────────────────────────────────────────────────────────┘
"""
from typing import Any, Dict, List, TypedDict, Optional, Literal
from copy import deepcopy
from langgraph.graph import StateGraph, END
from langchain_core.runnables import Runnable, RunnableConfig

from .BasicAgent import BasicAgent


class ScenarioCard(TypedDict):
    scenario_id: str # Unique identifier for this evaluation scenario
    scenario_info: str


class AttackState(TypedDict):
    """State for the multi-turn attack workflow.

    This state flows through all agents in the LangGraph workflow,
    carrying the context and results of each attack iteration.
    """
    # ========== Core Attack Parameters ==========
    intention: str  # The objective to achieve
    original_prompt: str  # Original prompt to attack
    wrapped_prompt: Optional[str]  # Generated attack prompt

    # ========== Iteration Control ==========
    max_iterations: int  # Maximum attack iterations
    current_iteration: int  # Current iteration counter

    # ========== Judgment and Evaluation ==========
    target_response: Optional[str]  # Response from target agent
    attack_is_success: bool  # Whether the current attack is successful

    # -------- Static (read-only after init) --------
    scenario: ScenarioCard # Static scenario specification for this run.

    # -------- Iteration control (dynamic) --------
    current_stage: str                # functional role of this turn (context_establishment, relationship_building, constraint_induction, escalation)
    experiment_config: Dict[str, Any] # run-level design config (full | prefix_2x2 | knockout)

    # -------- Per-turn artifacts (dynamic) --------
    planner_output: Dict[str, Any]    # selected next_stage, iv_updates, constraints
    ivs_activated: List[str]          # Backward-compatible alias of selected_dimensions
    strategy_selection: Dict[str, Any]  # Full strategy payload from IV recommender
    selected_dimensions: List[str]    # Selected factors for current stage
    strategy_plan: Dict[str, Any]     # Factor-level realization plan
    manipulation_check: Dict[str, Any]  # Raw manipulation-check payload
    judge_result: Dict[str, Any]      # turn validity, IV realization, target behavior tags, SSR observation, and diagnostic flags
    stage_alignment: int              # manipulation-check stage alignment (1-5)
    stage_leakage: int                # manipulation-check leakage score (1-5)
    manip_check_pass: bool            # whether manipulation check passed

    # ========== History and Logging ==========
    attack_history: List[Dict[str, Any]] # History of all attack attempts
    subgraph_optimize_history: List[Dict[str, Any]]  # History of all optimization attempts


class TargetAgent(BasicAgent):
    """The target agent that receives attack prompts.

    This is the model/system being attacked. It processes the wrapped
    prompts and generates responses that are then evaluated.
    """

    def __init__(
        self,
        agent: Runnable,
        system_prompt: str = "",
        session_id: str = "target_session",
        **basic_agent_kwargs: Any,
    ):
        """Initialize TargetAgent.

        Args:
            agent: The underlying language model
            system_prompt: System prompt to guide agent behavior
            session_id: Unique identifier for conversation history
            **basic_agent_kwargs: Optional BasicAgent retry/backoff args
        """
        super().__init__(agent, system_prompt, session_id, **basic_agent_kwargs)

    def __call__(self, state: AttackState) -> AttackState:
        """Process the wrapped prompt and return response.
        Flow:
        Generator -> wrapped_prompt -> TargetAgent -> target_response

        Args:
            state: Current attack state with wrapped_prompt

        Returns:
            Updated state with target_response
        """
        # Prepare input for the agent
        input_data = {"question": state['wrapped_prompt']}

        # Invoke the underlying agent through BasicAgent
        response = super().__call__(input_data, hidden=False)  # hidden=True to avoid logging raw responses in callbacks

        # Extract response content
        state['target_response'] = response.content if hasattr(response, 'content') else str(response)

        return state


class PlannerAgent(BasicAgent):
    """Manages attack strategy and progress.

    Analyzes current state and provides strategic guidance for
    the next attack iteration.
    """

    def __init__(
        self,
        agent: Runnable,
        system_prompt: str = "",
        session_id: str = "planner_session",
        **basic_agent_kwargs: Any,
    ):
        """Initialize PlannerAgent.

        Args:
            agent: The underlying language model
            system_prompt: System prompt to guide planning behavior
            session_id: Unique identifier for conversation history
            **basic_agent_kwargs: Optional BasicAgent retry/backoff args
        """
        super().__init__(agent, system_prompt, session_id, **basic_agent_kwargs)

    def _format_question(self, state: AttackState) -> str:
        """Format the planning question for the LLM.

        Args:
            state: Current attack state

        Returns:
            Formatted question string
        """
        # TODO: Analyze history, and then Build planning prompt
        raise NotImplementedError

    def _parse_guidance(self, response_text: str) -> list:
        """Parse LLM response to extract strategic guidance.

        Args:
            response_text: The LLM's planning response

        Returns:
            Extracted strategic guidance list
        """
        # TODO
        raise NotImplementedError

    def __call__(self, state: AttackState) -> AttackState:
        """Plan the next attack step.
        Flow:
        AttackState -> PlannerAgent -> planner_guidance -> PromptGeneratorAgent

        Args:
            state: Current attack state including history

        Returns:
            Updated state with strategic guidance
        """
        # Build planning question
        question = self._format_question(state)

        # Invoke the underlying agent through BasicAgent
        response = super().__call__({"question": question})

        # Extract response content
        resp_content = response.content if hasattr(response, 'content') else str(response)

        # Parse and update state with strategic guidance
        planner_output = self._parse_guidance(resp_content)
        state["planner_output"] = planner_output

        return state


class PromptGeneratorAgent(BasicAgent):
    """Creates wrapped prompts.

    Generates prompts based on the strategic guidance
    from the PlannerAgent.
    """

    def __init__(
        self,
        agent: Runnable,
        system_prompt: str = "",
        session_id: str = "generator_session",
        **basic_agent_kwargs: Any,
    ):
        """Initialize PromptGeneratorAgent.

        Args:
            agent: The underlying language model
            system_prompt: System prompt to guide generation behavior
            session_id: Unique identifier for conversation history
            **basic_agent_kwargs: Optional BasicAgent retry/backoff args
        """
        super().__init__(agent, system_prompt, session_id, **basic_agent_kwargs)

    def _format_question(self, state: AttackState) -> str:
        """Format the generation question for the LLM.

        Args:
            state: Current attack state

        Returns:
            Formatted question string
        """
        # TODO
        raise NotImplementedError

    def _parse_wrapped_prompt(self, response_text: str) -> str:
        """Parse LLM response to extract wrapped prompt.

        Args:
            response_text: The LLM's generation response

        Returns:
            Clean wrapped prompt string
        """
        # TODO
        raise NotImplementedError

    def __call__(self, state: AttackState) -> AttackState:
        """Generate a wrapped attack prompt.
        Flow:
        Planner -> planner_guidance -> PromptGeneratorAgent -> wrapped_prompt -> Target

        Args:
            state: Current attack state with strategic guidance

        Returns:
            Updated state with wrapped_prompt
        """
        # Build generation question
        question = self._format_question(state)

        # Invoke the underlying agent through BasicAgent
        response = super().__call__({"question": question})

        # Extract response content
        resp_content = response.content if hasattr(response, 'content') else str(response)

        # Parse and update state with wrapped prompt
        state["wrapped_prompt"] = self._parse_wrapped_prompt(resp_content)

        return state


class JudgementAgent(BasicAgent):
    """Evaluates attack success.

    Analyzes the target agent's response and determines if the
    attack was successful or if more iterations are needed.
    """

    def __init__(
        self,
        agent: Runnable,
        system_prompt: str = "",
        session_id: str = "judgement_session",
        **basic_agent_kwargs: Any,
    ):
        """Initialize JudgementAgent.

        Args:
            agent: The underlying language model
            system_prompt: System prompt to guide evaluation behavior
            session_id: Unique identifier for conversation history
            **basic_agent_kwargs: Optional BasicAgent retry/backoff args
        """
        super().__init__(agent, system_prompt, session_id, **basic_agent_kwargs)

    def _format_question(self, state: AttackState) -> str:
        """Format the evaluation question for the LLM.

        Args:
            state: Current attack state

        Returns:
            Formatted question string
        """
        # TODO
        raise NotImplementedError

    def _parse_judgement(self, response_text: str) -> bool:
        """Parse LLM response to determine if attack succeeded.

        Args:
            response_text: The LLM's judgment response text

        Returns:
            True if attack succeeded, False otherwise
        """
        # TODO
        raise NotImplementedError

    def __call__(self, state: AttackState) -> AttackState:
        """Judge the current attack result.
        Flow:
        Target -> target_response -> JudgementAgent -> attack_is_success -> _should_continue()

        Args:
            state: Current attack state with target_response

        Returns:
            Updated state with attack_is_success flag
        """
        current_iteration = state.get("current_iteration", 1)
        print(f"          [Iteration {current_iteration}] Judging response...", flush=True)

        # Build evaluation question from state
        question = self._format_question(state)

        # Invoke the underlying agent through BasicAgent
        response = super().__call__({"question": question})

        # Extract response content
        resp_content = response.content if hasattr(response, 'content') else str(response)

        # Parse judgment and update state
        state["attack_is_success"] = self._parse_judgement(resp_content)

        # Display judge result summary
        judge_score = state.get("judge_result", {}).get("judgement_score", "N/A")
        print(f"          [Iteration {current_iteration}] Judge score: {judge_score}, Success: {state['attack_is_success']}", flush=True)

        # Record in history
        state["attack_history"].append(deepcopy(state))

        # Increment iteration counter
        state["current_iteration"] = state.get("current_iteration", 1) + 1

        return state


class AttackWorkflow:
    """Orchestrates multi-turn attack workflow using LangGraph.

    Workflow flow:
        Planner -> Generator -> Target -> Judgement -> END

    The workflow continues until:
    - Attack succeeds (attack_is_success = True)
    - Max iterations reached
    """

    def __init__(
        self,
        target_agent: TargetAgent,
        planner_agent: PlannerAgent,
        generator_agent: PromptGeneratorAgent,
        judge_agent: JudgementAgent,
        max_iterations: int = 5,
    ):
        """Initialize the attack workflow.

        Args:
            target_agent: The target agent to be attacked
            planner_agent: Agent for planning strategy
            generator_agent: Agent for generating attack prompts
            judge_agent: Agent for evaluating success
            max_iterations: Maximum number of attack iterations
        """
        self.max_iterations = max_iterations
        self.target_agent = target_agent
        self.planner = planner_agent
        self.generator = generator_agent
        self.judgement = judge_agent
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow.

        Returns:
            Compiled StateGraph workflow
        """
        workflow = StateGraph(AttackState)

        # Add agent nodes
        workflow.add_node("planner", self.planner)
        workflow.add_node("generator", self.generator)
        workflow.add_node("target", self.target_agent)
        workflow.add_node("judgement", self.judgement)

        # Define workflow edges
        workflow.set_entry_point("planner")  # Start at planner
        workflow.add_edge("planner", "generator")  # planner -> generator
        workflow.add_edge("generator", "target")  # generator -> target
        workflow.add_edge("target", "judgement")  # target -> judgement

        # Conditional edge: continue loop or end
        workflow.add_conditional_edges(
            "judgement",
            self._should_continue,
            {"continue": "planner", "end": END}
        )

        return workflow.compile()

    def _should_continue(self, state: AttackState) -> str:
        """Determine if workflow should continue or end.

        This is a pure conditional function that checks state without modification.
        The iteration counter is incremented in JudgementAgent.__call__().

        Args:
            state: Current attack state

        Returns:
            "continue" if more iterations needed, "end" otherwise
        """
        # Success: end workflow
        if state.get("attack_is_success", False):
            return "end"

        # Max iterations reached: end workflow
        if state.get("current_iteration", 1) >= state.get("max_iterations", 5):
            return "end"

        # Otherwise, continue to next iteration
        return "continue"

    def run(
        self,
        intention: str,
        scenario: ScenarioCard,
        original_prompt: str,
        experiment_config: Dict[str, Any],
        config: RunnableConfig | None = None,
    ) -> AttackState:
        """Run the attack workflow.

        Args:
            intention: The objective to achieve
            scenario: Minimal scenario spec (static; read-only in-run)
            original_prompt: The original prompt to attack
            config: Optional config dict for LangGraph (e.g., for callbacks)

        Returns:
            Final attack state with complete history
        """
        # Initialize state
        initial_state: AttackState = {
            # Basic AttackState
            "intention": intention,
            "original_prompt": original_prompt,
            "wrapped_prompt": "",
            "max_iterations": self.max_iterations,
            "current_iteration": 0,
            "target_response": "",
            "attack_is_success": False,

            # Custom
            "scenario": scenario,
            "current_stage": "context_establishment",
            "experiment_config": experiment_config,
            "planner_output": {},
            "ivs_activated": [],
            "strategy_selection": {},
            "selected_dimensions": [],
            "strategy_plan": {},
            "manipulation_check": {},
            "judge_result": {},
            "stage_alignment": 1,
            "stage_leakage": 5,
            "manip_check_pass": False,

            "attack_history": [],
            "subgraph_optimize_history": [],
        }

        # Execute workflow
        print(f"        Starting workflow execution (max {self.max_iterations} iterations)...", flush=True)
        result = self.workflow.invoke(initial_state, config=config)
        iterations_completed = result.get("current_iteration", 0)
        print(f"        Workflow completed after {iterations_completed} iterations", flush=True)
        return result
