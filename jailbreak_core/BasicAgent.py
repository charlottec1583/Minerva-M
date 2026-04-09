"""Basic Agent wrapper for multi-turn conversations with message history.

This module provides a simplified wrapper around AgentNode for easy use in
attack workflows and other agent-based systems.
"""
import time
import re
import uuid
import random
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.messages import BaseMessage, AIMessage
from openai import APITimeoutError, APIConnectionError, APIStatusError, OpenAIError

from wagent.agent_flow import AgentInput, AgentNode


class BasicAgent:
    """Basic Agent wrapper with message history support.

    This class provides a simplified interface to AgentNode, making it easy
    to create agents with persistent conversation history across sessions.
    """

    def __init__(
        self,
        agent: Runnable,
        system_prompt: str = "",
        session_id: str = "default_session",
        max_retry_num: int = 3,
        invoke_delay: float = 1.0,
        invoke_delay_max: float = 15.0,
        retry_backoff_factor: float = 2.0,
        retry_jitter: float = 0.15,
    ):
        """Initialize BasicAgent with an LLM and system prompt.

        Args:
            agent: The underlying language model (ChatOpenAI, AgentOpenAI, etc.)
            system_prompt: System prompt to guide agent behavior
            session_id: Unique identifier for conversation history
            max_retry_num: Maximum number of retry attempts for API errors (default: 3)
            invoke_delay: Delay in seconds between retry attempts (default: 1.0)
            invoke_delay_max: Maximum delay in seconds for exponential backoff (default: 15.0)
            retry_backoff_factor: Backoff multiplier for each retry (default: 2.0)
            retry_jitter: Jitter ratio [0,1] to avoid synchronized retries (default: 0.15)
        """
        # Create AgentNode with the LLM and system prompt
        self.agent_node = AgentNode(
            agent=agent,
            system_prompt=system_prompt,
        )

        self._session_id = session_id
        self._default_config = RunnableConfig(
            configurable={"session_id": session_id}
        )

        # Retry configuration
        self.max_retry_num = max_retry_num
        self.invoke_delay = invoke_delay
        self.invoke_delay_max = max(invoke_delay, invoke_delay_max)
        self.retry_backoff_factor = max(1.0, retry_backoff_factor)
        self.retry_jitter = min(max(0.0, retry_jitter), 1.0)

    def _compute_retry_delay(self, attempt_idx: int, error: Exception | None = None) -> float:
        """Compute retry delay with exponential backoff and optional jitter.

        attempt_idx is 0-based, so the first retry uses invoke_delay.
        """
        base = self.invoke_delay * (self.retry_backoff_factor ** attempt_idx)
        delay = min(base, self.invoke_delay_max)

        # Add bounded symmetric jitter to reduce retry stampede.
        if self.retry_jitter > 0:
            jitter_span = delay * self.retry_jitter
            delay = max(0.0, delay + random.uniform(-jitter_span, jitter_span))
        return delay

    def _extract_response_content(self, response_text: str) -> str:
        """Extract content from response, skipping <think> tags.

        Handles formats:
        1. <think>...</think><response>...</response> - extracts content from <response>
        2. <think>...</think>content - extracts content after </think>
        3. <response>...</response> - extracts content from <response>
        4. Plain content - returns as is

        Args:
            response_text: Raw response text that may contain <think> and/or <response> tags

        Returns:
            Extracted response content without tags
        """
        # First, remove <think>...</think> tags and their content
        # Pattern matches <think>optional content</think> (case insensitive)
        text_without_think = re.sub(
            r'<think>.*?</think>',
            '',
            response_text,
            flags=re.DOTALL | re.IGNORECASE
        )

        # Then, try to extract content from <response> tags if present
        pattern = r'<response>\s*(.*?)\s*</response>'
        match = re.search(pattern, text_without_think, re.DOTALL | re.IGNORECASE)

        if match:
            return match.group(1).strip()

        # If no <response> tags found, return the text without <think> tags
        return text_without_think.strip()

    def _process_response_content(self, response: BaseMessage) -> BaseMessage:
        """Process response content by extracting from <response> tags if present.

        Args:
            response: The response message to process

        Returns:
            The response with content extracted from <response> tags if present
        """
        if hasattr(response, "content"):
            raw_content = response.content
            extracted_content = self._extract_response_content(raw_content)
            response.content = extracted_content
        return response

    def __call__(
        self,
        input_data: AgentInput,
        no_past_history: bool = False,
        hidden: bool = False,
        summary_agent: AgentNode = None,
        summary_prompt_preamble: str = "Please summarize the following conversation concisely:\n\n",
        summary_skip_recent_msg_num: int | float = 0.1,
        summary_skip_decay_factor: float = 0.9,
    ) -> BaseMessage:
        """Invoke the agent with message history support.

        This is the main interface for interacting with the agent.
        Conversation history is automatically maintained across calls.

        Args:
            input_data: Input dictionary with "question" key
            no_past_history: If True, bypasses conversation history for this call.
                The interaction will not be recorded in the session history and
                will not have access to previous messages. Useful for isolated
                queries or private interactions that shouldn't influence future
                conversations (default: False)
            hidden: If True, run without affecting the original session history.
                   The agent sees a copy of the history but updates are discarded.
            summary_agent: Optional AgentNode to use for generating memory summaries
                when context length is exceeded. If not provided, memory summarization
                will be skipped.
            summary_prompt_preamble: The prompt prefix to add before the history text
                when requesting a summary (default: "Please summarize the following
                conversation concisely:\n\n")
            summary_skip_recent_msg_num: Controls how much history to skip when summarizing
                on context length exceeded errors.
                - If float (0.0-1.0): Represents the **skip ratio** of history
                  (e.g., 0.1 means skip the most recent 10% of history for summarization).
                - If int: Represents the exact **number of recent messages** to skip
                  (e.g., 10 means skip the most recent 10 messages).
                The skipped recent messages are preserved as-is, while older messages
                are summarized. **Counts the sum of human and AI messages**.
            summary_skip_decay_factor: Decay factor (0.0-1.0) for progressive history reduction
                on each retry. Higher values (e.g., 0.9) reduce more slowly, preserving more
                context across retries. (default: 0.9)

        Returns:
            BaseMessage: The agent's response

        Raises:
            RuntimeError: If all retry attempts fail

        Example:
            >>> agent = BasicAgent(agent, system_prompt="You are helpful.", session_id="session_1")
            >>> response = agent({"question": "Hello!"})
            >>> print(response.content)
            >>> # Hidden interaction - not recorded in history
            >>> response = agent({"question": "Secret prompt"}, hidden=True)
        """
        # Track response and error across retry attempts
        response = None
        error = None

        # Use a temporary session for no_past_history interactions
        # When no_past_history=True, set session_id to None to bypass history
        config = self._default_config.copy()
        if no_past_history:
            config['configurable'] = config['configurable'] or {}
            config['configurable']["session_id"] = None

        # Validate and normalize summary parameters
        # Float values represent ratios and must be in [0, 1] range
        if isinstance(summary_skip_recent_msg_num, float):
            summary_skip_recent_msg_num = min(max(summary_skip_recent_msg_num, 0), 1)
        elif isinstance(summary_skip_recent_msg_num, int):
            summary_skip_recent_msg_num = max(summary_skip_recent_msg_num, 1)
        else:
            raise ValueError(
                f"summary_skip_recent_msg_num must be int or float, "
                f"got {type(summary_skip_recent_msg_num).__name__}"
            )

        # Clamp decay factor to [0, 1] to ensure it behaves correctly
        summary_skip_decay_factor = min(max(summary_skip_decay_factor, 0), 1)

        for attempt in range(self.max_retry_num):
            try:
                if error is None or not is_context_length_exceeded_error(error):
                    # Invoke the agent node with input and config
                    response = self.agent_node.invoke(input_data, config=config, hidden=hidden)
                    return self._process_response_content(response)
                else:
                    # Calculate current skip with decay factor
                    decay_power = max(0, attempt - 1)
                    if isinstance(summary_skip_recent_msg_num, float):
                        # Float mode: summary_skip_recent_msg_num is the skip ratio.
                        # Example: skip_ratio=0.1, decay=0.9
                        #   - 1st retry: 1 - (1-0.1) * 0.9^0 = 0.1 (skip 10% of history)
                        #   - 2nd retry: 1 - (1-0.1) * 0.9^1 = 0.19 (skip 19% of history)
                        current_skip = 1 - (1 - summary_skip_recent_msg_num) * (summary_skip_decay_factor ** decay_power)
                        # Clamp to [0, 1] for safety (handles edge cases with floating-point precision)
                        current_skip = min(max(current_skip, 0.0), 1.0)
                    elif isinstance(summary_skip_recent_msg_num, int):
                        # Integer mode: summary_skip_recent_msg_num is the base skip count.
                        # Progressive reduction: current_skip = base * decay_factor^decay_power
                        # Example: base=10, decay=0.9
                        #   - 1st retry: 10 * 0.9^0 = 10 messages skipped
                        #   - 2nd retry: 10 * 0.9^1 = 9 messages skipped
                        current_skip = int(summary_skip_recent_msg_num * summary_skip_decay_factor ** decay_power)
                        current_skip = max(0, current_skip)

                    # Step 1: Get history text for summarization (skipping recent n messages)
                    # This preserves recent context while summarizing older messages
                    history_for_summary = self.agent_node.get_history_for_summary(
                        config, skip_recent_messages=current_skip
                    )

                    # Step 2: Determine which agent to use for summarization
                    # If no summary_agent provided, use self.agent_node as fallback
                    summary_agent_to_use = summary_agent if summary_agent else self.agent_node

                    # Step 3: Generate summary from the conversation history
                    summary_context = generate_summary(
                        history_for_summary,
                        summary_agent_to_use,
                        summary_prompt_preamble,
                    )

                    # Retry the invocation with summary and limited history
                    response = self.agent_node.invoke(
                        input_data,
                        config=config,
                        hidden=hidden,
                        max_history_messages=current_skip,
                        summary_text=summary_context,
                    )
                    return self._process_response_content(response)

            except (APITimeoutError, APIConnectionError, APIStatusError, OpenAIError) as e:
                error = e
                print(f"Attempt {attempt+1}/{self.max_retry_num}: {type(e).__name__} - {e}")

            except Exception as e:
                error = e
                print(f"Attempt {attempt+1}/{self.max_retry_num}, Unexpected error: {e}")
                # # Print full traceback for debugging
                # import traceback
                # traceback.print_exc()

            # Wait before retrying (skip delay after last attempt)
            if attempt < self.max_retry_num - 1:
                delay = self._compute_retry_delay(attempt, error)
                print(f"Retrying in {delay:.2f}s...")
                time.sleep(delay)

        # All retry attempts failed. Return an error response instead of raising.
        # This allows batch testing to continue instead of stopping on errors.
        print(f"All {self.max_retry_num} attempts failed.")

        # Create an AIMessage with error information
        error_content = f"[AGENT ERROR] Failed after {self.max_retry_num} retry attempts."
        if error is not None:
            error_content += f" Last error: {type(error).__name__}: {error}"

        # Return AIMessage with error info embedded
        error_response = AIMessage(content=error_content)
        # Attach error metadata for detection by upper layers
        error_response.error = error
        error_response.error_type = type(error).__name__ if error else None

        return self._process_response_content(error_response)

    def reset_history(self) -> None:
        """Reset the conversation history for this agent's session.

        Useful for starting a fresh conversation without creating a new agent.

        Example:
            >>> agent.reset_history()
            >>> response = agent({"question": "New conversation start"})
        """
        if self._session_id in self.agent_node.sessions:
            self.agent_node.sessions[self._session_id].clear()

    def get_history(self) -> list:
        """Get the conversation history for this agent's session.

        Returns:
            List of Message objects representing the conversation history

        Example:
            >>> history = agent.get_history()
            >>> for msg in history:
            ...     print(f"{msg.type}: {msg.content}")
        """
        if self._session_id in self.agent_node.sessions:
            history = self.agent_node.sessions[self._session_id]
            return history.messages
        return []

    def change_session(self, new_session_id: str) -> None:
        """Change to a different session.

        This allows the same agent to manage multiple conversations.

        Args:
            new_session_id: The new session identifier

        Example:
            >>> agent.change_session("user_2_session")
            >>> response = agent({"question": "Hello user 2"})
        """
        self.session_id = new_session_id

    @property
    def session_id(self) -> str:
        """Get the current session ID.

        Example:
            >>> current_session = agent.session_id
            >>> print(current_session)  # "default_session"
        """
        return self._session_id

    @session_id.setter
    def session_id(self, new_session_id: str) -> None:
        """Set a new session ID and update the default config.

        Example:
            >>> agent.session_id = "new_session"
        """
        self._session_id = new_session_id
        self._default_config = RunnableConfig(
            configurable={"session_id": new_session_id}
        )

    @property
    def system_prompt(self) -> str:
        """Get the current system prompt.

        Example:
            >>> prompt = agent.system_prompt
            >>> print(prompt)
        """
        return self.agent_node.system_prompt

    @system_prompt.setter
    def system_prompt(self, new_prompt: str) -> None:
        """Set a new system prompt.

        Note: This will update the prompt for future invocations but
        won't affect existing conversation history.

        Example:
            >>> agent.system_prompt = "You are now a strict assistant."
        """
        # Access private attribute to update prompt
        object.__setattr__(
            self.agent_node,
            'system_prompt',
            new_prompt
        )


def is_context_length_exceeded_error(error: Exception) -> bool:
    """Check if the error is due to context length being exceeded.

    Args:
        error: The exception to check

    Returns:
        True if the error indicates context length exceeded
    """
    error_str = str(error).lower()
    context_length_keywords = [
        "context length",
        "maximum context length",
        "max_tokens",
        "token limit",
        "too many tokens",
        # Error Message: {"error":{"code":"invalid_argument","message":"input characters limit
        # is 256000","type":"invalid_request_error"},"id":"as-r5s9hmg775"}{"code":20000,"msg":
        # "Unknown error []"}
        "characters limit",
    ]
    return any(keyword in error_str for keyword in context_length_keywords)


def generate_summary(
    history_for_summary: str,
    summary_agent: AgentNode,
    summary_prompt_preamble: str = "Please summarize the following conversation concisely:\n\n",
) -> str:
    """Generate a summary of the conversation history.

    This function uses an LLM agent to create a condensed summary of the
    conversation history, which can be used to reduce token usage when the
    context length is exceeded.

    Args:
        history_for_summary: The conversation history text to summarize
        summary_agent: AgentNode (or composed agent chain) to use for generating
            the summary. Can be a single agent or a chain like: summary_agent | basic_agent
        summary_prompt_preamble: The prompt prefix to add before the history text
            when requesting a summary (default: "Please summarize the following
            conversation concisely:\n\n")

    Returns:
        Generated summary text. Returns empty string if summary generation fails
        or if history_for_summary is empty.
    """
    if not history_for_summary or history_for_summary.strip() == "":
        return ""

    # Build the summary prompt with preamble and history
    summary_prompt = summary_prompt_preamble + history_for_summary

    # Create a unique throwaway session ID for this summarization call.
    # Using a temporary session ensures the summary generation doesn't
    # interfere with or become part of the main conversation history.
    temp_session_id = f"summary_tmp_{uuid.uuid4().hex}"
    temp_summary_config = RunnableConfig(configurable={"session_id": temp_session_id})
    try:
        summary_response = summary_agent.invoke(summary_prompt, config=temp_summary_config)
        summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
        return summary.strip()
    except Exception as e:
        print(f"Failed to generate summary: {e}")
        return ""
