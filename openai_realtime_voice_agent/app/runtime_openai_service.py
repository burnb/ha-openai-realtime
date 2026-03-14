"""Runtime-safe OpenAI Realtime service helpers for a shared pipeline."""

from typing import Optional

from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService


class RuntimeOpenAIRealtimeLLMService(OpenAIRealtimeLLMService):
    """Adds explicit runtime session reset helpers for a long-lived pipeline."""

    async def reset_runtime_session(
        self,
        *,
        context: Optional[LLMContext] = None,
        session_properties: Optional[events.SessionProperties] = None,
    ) -> None:
        """Reconnect the OpenAI realtime websocket with a specific runtime context."""
        if session_properties is not None:
            self._settings.session_properties = session_properties

        self._context = context or LLMContext()
        self._messages_added_manually = {}
        self._pending_function_calls = {}
        self._completed_tool_calls = set()
        self._current_assistant_response = None
        self._current_audio_response = None
        self._run_llm_when_api_session_ready = False
        self._llm_needs_conversation_setup = True

        await self.reset_conversation()

    async def close_runtime_session(self) -> None:
        """Close the current OpenAI realtime websocket and clear local state."""
        self._context = LLMContext()
        self._messages_added_manually = {}
        self._pending_function_calls = {}
        self._completed_tool_calls = set()
        self._current_assistant_response = None
        self._current_audio_response = None
        self._run_llm_when_api_session_ready = False
        self._llm_needs_conversation_setup = True

        await self._disconnect()