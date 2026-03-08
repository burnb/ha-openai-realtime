"""Guard against overlapping OpenAI Realtime response creation."""

import asyncio
import json
import logging
import time

from pipecat.adapters.services.open_ai_realtime_adapter import OpenAIRealtimeLLMAdapter
from pipecat.frames.frames import LLMFullResponseStartFrame
from pipecat.services.openai.realtime import events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from websockets.exceptions import ConnectionClosed


logger = logging.getLogger(__name__)

MAX_OPENAI_SESSION_AGE_SECONDS = 55 * 60


class GuardedOpenAIRealtimeLLMService(OpenAIRealtimeLLMService):
    """Serialize response creation to avoid overlap errors from the Realtime API."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._response_guard_lock = asyncio.Lock()
        self._session_refresh_lock = asyncio.Lock()
        self._response_in_progress = False
        self._deferred_response_create = False
        self._session_started_monotonic: float | None = None

    async def _connect(self):
        await super()._connect()
        if self._websocket:
            self._session_started_monotonic = time.monotonic()

    def _is_session_age_expired(self) -> bool:
        return (
            self._session_started_monotonic is not None
            and time.monotonic() - self._session_started_monotonic >= MAX_OPENAI_SESSION_AGE_SECONDS
        )

    def _is_session_duration_close(self, exc: Exception) -> bool:
        if not isinstance(exc, ConnectionClosed):
            return False
        error_text = str(exc).lower()
        return "maximum duration" in error_text and "60 minutes" in error_text

    def _in_receive_task(self) -> bool:
        try:
            return asyncio.current_task() is self._receive_task
        except RuntimeError:
            return False

    async def _ensure_session_available(self, reason: str, force: bool = False) -> bool:
        if self._disconnecting or self._in_receive_task():
            return False

        async with self._session_refresh_lock:
            needs_refresh = force or self._websocket is None or self._is_session_age_expired()
            if not needs_refresh:
                return False

            logger.warning("Refreshing OpenAI Realtime session: %s", reason)
            await self.reset_conversation()
            return True

    async def _defer_response_create(self, reason: str) -> None:
        async with self._response_guard_lock:
            self._deferred_response_create = True
        logger.info("Deferring OpenAI response.create: %s", reason)

    async def _release_response_slot(self) -> bool:
        async with self._response_guard_lock:
            should_create = self._deferred_response_create
            self._response_in_progress = False
            self._deferred_response_create = False
            return should_create

    async def _mark_response_in_progress(self) -> bool:
        async with self._response_guard_lock:
            if self._response_in_progress:
                self._deferred_response_create = True
                return False
            self._response_in_progress = True
            return True

    async def send_client_event(self, event: events.ClientEvent):
        """Guard direct response.create events sent outside of _create_response."""
        await self._ensure_session_available("proactive session rotation before send")

        if isinstance(event, events.ResponseCreateEvent):
            reserved = await self._mark_response_in_progress()
            if not reserved:
                logger.info("Deferring direct OpenAI response.create until current response completes")
                return

            try:
                await super().send_client_event(event)
            except Exception:
                await self._release_response_slot()
                raise
            return

        await super().send_client_event(event)

    async def _receive_task_handler(self):
        try:
            async for message in self._websocket:
                evt = events.parse_server_event(message)
                if evt.type == "session.created":
                    await self._handle_evt_session_created(evt)
                elif evt.type == "session.updated":
                    await self._handle_evt_session_updated(evt)
                elif evt.type == "response.output_audio.delta":
                    await self._handle_evt_audio_delta(evt)
                elif evt.type == "response.output_audio.done":
                    await self._handle_evt_audio_done(evt)
                elif evt.type == "conversation.item.added":
                    await self._handle_evt_conversation_item_added(evt)
                elif evt.type == "conversation.item.done":
                    await self._handle_evt_conversation_item_done(evt)
                elif evt.type == "conversation.item.input_audio_transcription.delta":
                    await self._handle_evt_input_audio_transcription_delta(evt)
                elif evt.type == "conversation.item.input_audio_transcription.completed":
                    await self.handle_evt_input_audio_transcription_completed(evt)
                elif evt.type == "conversation.item.retrieved":
                    await self._handle_conversation_item_retrieved(evt)
                elif evt.type == "response.done":
                    await self._handle_evt_response_done(evt)
                elif evt.type == "input_audio_buffer.speech_started":
                    await self._handle_evt_speech_started(evt)
                elif evt.type == "input_audio_buffer.speech_stopped":
                    await self._handle_evt_speech_stopped(evt)
                elif evt.type == "response.output_text.delta":
                    await self._handle_evt_text_delta(evt)
                elif evt.type == "response.output_audio_transcript.delta":
                    await self._handle_evt_audio_transcript_delta(evt)
                elif evt.type == "response.function_call_arguments.done":
                    await self._handle_evt_function_call_arguments_done(evt)
                elif evt.type == "error":
                    if not await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
                        if evt.error.code == "response_cancel_not_active":
                            logger.debug("%s %s", self, evt.error.message)
                        elif evt.error.code == "conversation_already_has_active_response":
                            await self._defer_response_create(evt.error.message)
                        else:
                            await self._handle_evt_error(evt)
                            return
        except ConnectionClosed as exc:
            if not self._disconnecting and not self._is_session_duration_close(exc):
                await self.push_error(error_msg=f"Error receiving client event: {exc}", exception=exc)
            elif self._is_session_duration_close(exc):
                logger.warning("OpenAI Realtime session reached maximum duration; next outbound event will reconnect")
        finally:
            if not self._disconnecting:
                self._api_session_ready = False
                self._websocket = None
                self._receive_task = None

    async def _handle_evt_response_done(self, evt):
        try:
            await super()._handle_evt_response_done(evt)
        finally:
            should_create = await self._release_response_slot()

        if should_create:
            logger.info("Flushing deferred OpenAI response.create after response.done")
            await self._create_response()

    async def _disconnect(self):
        try:
            await super()._disconnect()
        finally:
            async with self._response_guard_lock:
                self._response_in_progress = False
                self._deferred_response_create = False
            self._session_started_monotonic = None

    async def _ws_send(self, realtime_message):
        try:
            if not self._disconnecting and self._websocket:
                await self._websocket.send(json.dumps(realtime_message))
        except ConnectionClosed as exc:
            if self._disconnecting:
                return
            if not self._is_session_duration_close(exc):
                await self.push_error(error_msg=f"Error sending client event: {exc}", exception=exc)
                return

            logger.warning(
                "OpenAI Realtime session expired while sending %s; reconnecting and retrying",
                realtime_message.get("type", "unknown"),
            )
            self._api_session_ready = False
            self._websocket = None

            refreshed = await self._ensure_session_available(
                "session expired while sending client event",
                force=True,
            )
            if not refreshed or not self._websocket:
                if realtime_message.get("type") == "response.create":
                    await self._release_response_slot()
                await self.push_error(error_msg=f"Error sending client event: {exc}", exception=exc)
                return

            if realtime_message.get("type") == "response.create":
                await self._mark_response_in_progress()

            try:
                await self._websocket.send(json.dumps(realtime_message))
            except Exception as retry_exc:
                if realtime_message.get("type") == "response.create":
                    await self._release_response_slot()
                await self.push_error(
                    error_msg=f"Error sending client event after session refresh: {retry_exc}",
                    exception=retry_exc,
                )
        except Exception as exc:
            if self._disconnecting or not self._websocket:
                return
            await self.push_error(error_msg=f"Error sending client event: {exc}", exception=exc)

    async def _create_response(self):
        if not self._api_session_ready:
            await super()._create_response()
            return

        reserved = await self._mark_response_in_progress()
        if not reserved:
            logger.info("Deferring OpenAI response.create until current response completes")
            return

        try:
            adapter: OpenAIRealtimeLLMAdapter = self.get_llm_adapter()

            if self._llm_needs_conversation_setup:
                logger.debug(
                    "Setting up conversation on OpenAI Realtime LLM service with initial messages: %s",
                    adapter.get_messages_for_logging(self._context),
                )

                llm_invocation_params = adapter.get_llm_invocation_params(self._context)
                messages = llm_invocation_params["messages"]
                for item in messages:
                    evt = events.ConversationItemCreateEvent(item=item)
                    self._messages_added_manually[evt.item.id] = True
                    await super().send_client_event(evt)

                await self._send_session_update()
                self._llm_needs_conversation_setup = False

            logger.debug("Creating response")

            await self.push_frame(LLMFullResponseStartFrame())
            await self.start_processing_metrics()
            await self.start_ttfb_metrics()
            await super().send_client_event(
                events.ResponseCreateEvent(
                    response=events.ResponseProperties(
                        output_modalities=self._get_enabled_modalities()
                    )
                )
            )
        except Exception:
            await self._release_response_slot()
            raise