"""Main application entry point using Pipecat."""
import copy
import os
import sys
import asyncio
import json
import logging
from typing import Optional
import dotenv
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.transports.websocket.server import WebsocketServerTransport
from app.home_assistant_context import HomeAssistantContextService
from app.mcp_service import HomeAssistantMCPService
from app.runtime_openai_service import RuntimeOpenAIRealtimeLLMService
from app.disconnect_tool import (
    create_disconnect_tool_handler,
    get_disconnect_tool_definition,
    parse_disconnect_trigger_phrases,
)
from app.audio_recording_service import AudioRecordingService
from app.session_manager import SessionManager, approx_tokens_from_chars
from app.websocket_handler import WebSocketHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce verbosity of noisy loggers
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.INFO)

dotenv.load_dotenv()


class Application:
    """Main application class using Pipecat."""
    
    def __init__(self):
        """Initialize application."""
        self.pipeline: Optional[Pipeline] = None
        self.runner: Optional[PipelineRunner] = None
        self.websocket_handler: Optional[WebSocketHandler] = None
        self.websocket_transport: Optional[WebsocketServerTransport] = None
        self.openai_service: Optional[RuntimeOpenAIRealtimeLLMService] = None
        self.mcp_service: Optional[HomeAssistantMCPService] = None
        self.ha_context_service: Optional[HomeAssistantContextService] = None
        self.audio_recording_service: Optional[AudioRecordingService] = None
        self.session_manager: Optional[SessionManager] = None
        self.current_task: Optional[PipelineTask] = None
        self._pipeline_lock: Optional[asyncio.Lock] = None
        self._base_session_properties = None
        self._base_mcp_tools_schema = None
        self._base_all_tools: Optional[list[dict]] = None
        
    async def initialize(self) -> None:
        """Initialize all components."""
        # Get configuration from environment
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        openai_model = os.environ.get("OPENAI_MODEL", "gpt-realtime-mini")
        openai_voice = os.environ.get("OPENAI_VOICE", "ash")
        openai_sst_model = os.environ.get("OPENAI_SST_MODEL", "gpt-4o-mini-transcribe")
        openai_sst_language = os.environ.get("OPENAI_SST_LANGUAGE", "") or None
        websocket_port = int(os.environ.get("WEBSOCKET_PORT", "8080"))
        websocket_host = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
        
        # Get turn detection settings with defaults
        vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.5"))
        vad_prefix_padding_ms = int(os.environ.get("VAD_PREFIX_PADDING_MS", "300"))
        vad_silence_duration_ms = int(os.environ.get("VAD_SILENCE_DURATION_MS", "500"))
        
        # Get instructions with default
        instructions = os.environ.get("INSTRUCTIONS", "You are the Home Assistant Voice Agent and can control the Smart Home.")
        mcp_tool_filter = self._parse_mcp_tool_filter(os.environ.get("HA_MCP_TOOL_FILTER", ""))
        
        # Get recording setting (optional, defaults to false)
        enable_recording = os.environ.get("ENABLE_RECORDING", "false").lower() == "true"
        
        # Get session reuse timeout and initialize session manager
        session_reuse_timeout = float(os.environ.get("SESSION_REUSE_TIMEOUT_SECONDS", "300"))
        self.session_manager = SessionManager(reuse_timeout=session_reuse_timeout)
        logger.info(f"Session reuse timeout: {session_reuse_timeout} seconds")
        disconnect_trigger_phrases = parse_disconnect_trigger_phrases(
            os.environ.get("DISCONNECT_TRIGGER_PHRASES", "")
        )
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize Home Assistant MCP Service
        mcp_client = None
        try:
            supervisor_token = os.environ.get("LONGLIVED_TOKEN") or os.environ.get("SUPERVISOR_TOKEN")
            ha_mcp_url = os.environ.get("HA_MCP_URL", "http://supervisor/core/api/mcp")
            if supervisor_token:
                logger.info("Loading Home Assistant MCP tools...")
                self.mcp_service = HomeAssistantMCPService(
                    url=ha_mcp_url,
                    access_token=supervisor_token,
                    excluded_tools=mcp_tool_filter,
                )
                self.ha_context_service = HomeAssistantContextService(
                    mcp_url=ha_mcp_url,
                    access_token=supervisor_token,
                )
                mcp_client = await self.mcp_service.initialize()
                dynamic_instructions = await self._build_home_assistant_instructions(instructions)
                if dynamic_instructions != instructions:
                    logger.info("✅ Home Assistant exposed-entity context appended to instructions")
                instructions = dynamic_instructions
                logger.info("Full session instructions:\n%s", instructions)
                logger.info("✅ Home Assistant MCP Client initialized")
            else:
                logger.warning("⚠️ SUPERVISOR_TOKEN not set, skipping Home Assistant MCP integration")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Home Assistant MCP Client: {e}")
        
        # Initialize WebSocket handler
        self.websocket_handler = WebSocketHandler(
            host=websocket_host,
            port=websocket_port,
            session_manager=self.session_manager,
            audio_recording_service=self.audio_recording_service
        )
        self.websocket_transport = self.websocket_handler.create_transport()
        
        # Store configuration for session creation
        self.openai_api_key = openai_api_key
        self.openai_voice = openai_voice
        self.openai_model = openai_model
        self.openai_sst_model = openai_sst_model
        self.openai_sst_language = openai_sst_language
        self.vad_threshold = vad_threshold
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.disconnect_trigger_phrases = disconnect_trigger_phrases
        self.instructions = instructions
        self.mcp_client = mcp_client
        (
            self._base_session_properties,
            self._base_mcp_tools_schema,
            self._base_all_tools,
        ) = await self._build_session_configuration()
        
        # Initialize audio recording service (optional)
        self.audio_recording_service = AudioRecordingService(
            enable_recording=enable_recording,
            sample_rate=24000,
            chunk_duration_seconds=30,
            output_dir="recordings"
        )
        
        logger.info("✅ Application initialized - ready to accept WebSocket connections")

    async def _build_home_assistant_instructions(self, base_instructions: str) -> str:
        """Append exposed Home Assistant voice-assistant context to the base instructions."""
        if not self.ha_context_service:
            return base_instructions

        snapshot = await self.ha_context_service.fetch_snapshot()
        if not snapshot:
            return base_instructions

        return f"{base_instructions.rstrip()}{snapshot.to_instructions()}"

    def _parse_mcp_tool_filter(self, raw_value: str) -> Optional[list[str]]:
        """Parse comma-separated MCP tool names to exclude from the environment."""
        tool_names = [tool.strip() for tool in raw_value.split(",") if tool.strip()]
        if tool_names:
            logger.info("MCP tool exclude filter enabled: %s", tool_names)
            return tool_names
        return None

    def _log_session_payload_metrics(self, tools: list[dict], client_id: Optional[str]) -> None:
        """Log approximate prompt size contributors before session creation."""
        instructions_chars = len(self.instructions)
        tools_chars = len(json.dumps(tools, ensure_ascii=False))
        cached_metrics = {"messages": 0, "chars": 0, "approx_tokens": 0}
        if client_id and self.session_manager:
            cached_metrics = self.session_manager.get_cached_context_metrics(client_id)

        logger.info(
            "📊 Session payload estimate for %s: instructions=%s chars (~%s tokens), tools=%s chars (~%s tokens, %s tools), cached_context=%s messages / %s chars (~%s tokens)",
            client_id or "server",
            instructions_chars,
            approx_tokens_from_chars(instructions_chars),
            tools_chars,
            approx_tokens_from_chars(tools_chars),
            len(tools),
            cached_metrics["messages"],
            cached_metrics["chars"],
            cached_metrics["approx_tokens"],
        )
    
    def _build_pipeline_for_transport(self, transport: WebsocketServerTransport, client_id: str):
        """
        Build pipeline for a WebSocket transport connection.
        
        Args:
            transport: The WebSocket transport instance
            client_id: Unique identifier for the client device
        """
        # Ensure OpenAI service exists
        if self.openai_service is None:
            raise RuntimeError("OpenAI service must be created before building pipeline")
        
        # Use WebSocket handler to build pipeline
        self.pipeline, self.runner, self.current_task = self.websocket_handler.build_pipeline(
            transport=transport,
            openai_service=self.openai_service,
            client_id=client_id,
            activity_callback=self._update_session_activity
        )
    
    def _update_session_activity(self):
        """Update session activity timestamp (called by SessionActivityTracker)."""
        pass
    
    async def _build_session_configuration(self):
        """Build current session properties and tool registrations."""
        from pipecat.services.openai.realtime.events import (
            SessionProperties,
            AudioConfiguration,
            AudioInput,
            AudioOutput,
            TurnDetection,
            InputAudioTranscription,
            InputAudioNoiseReduction,
        )

        disconnect_tool_def = get_disconnect_tool_definition(self.disconnect_trigger_phrases)
        all_tools = [disconnect_tool_def]

        mcp_tools_schema = None
        if self.mcp_client:
            try:
                logger.info("🔧 Fetching MCP tool definitions...")
                mcp_tools_schema = await self.mcp_client.get_tools_schema()
                if self.mcp_service:
                    mcp_tools_schema = self.mcp_service.filter_tools_schema(mcp_tools_schema)

                for function_schema in mcp_tools_schema.standard_tools:
                    openai_tool = {
                        "type": "function",
                        "name": function_schema.name,
                        "description": function_schema.description,
                        "parameters": {
                            "type": "object",
                            "properties": function_schema.properties,
                            "required": function_schema.required,
                        },
                    }
                    all_tools.append(openai_tool)

                logger.info(f"✅ Fetched {len(mcp_tools_schema.standard_tools)} MCP tools")
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch MCP tool definitions: {e}")

        session_properties = SessionProperties(
            instructions=self.instructions,
            audio=AudioConfiguration(
                input=AudioInput(
                    turn_detection=TurnDetection(
                        type="server_vad",
                        threshold=self.vad_threshold,
                        prefix_padding_ms=self.vad_prefix_padding_ms,
                        silence_duration_ms=self.vad_silence_duration_ms,
                    ),
                    transcription=InputAudioTranscription(
                        model=self.openai_sst_model,
                        language=self.openai_sst_language,
                    ),
                    # noise_reduction=InputAudioNoiseReduction(type="far_field"),
                ),
                output=AudioOutput(voice=self.openai_voice),
            ),
            tools=all_tools,
        )

        return session_properties, mcp_tools_schema, all_tools

    async def _ensure_openai_service(self, client_id: Optional[str] = None):
        """Create the shared OpenAI service and refresh its session for a client.
        
        Args:
            client_id: Optional client ID for session management
        """
        if self._pipeline_lock is None:
            self._pipeline_lock = asyncio.Lock()
        
        async with self._pipeline_lock:
            session_properties = copy.deepcopy(self._base_session_properties)
            mcp_tools_schema = self._base_mcp_tools_schema
            all_tools = copy.deepcopy(self._base_all_tools) if self._base_all_tools else []
            self._log_session_payload_metrics(all_tools, client_id)

            if self.openai_service is None:
                logger.info(f"🔧 Creating session with {len(all_tools)} tools: {[tool.get('name', 'unknown') for tool in all_tools]}")
                self.openai_service = RuntimeOpenAIRealtimeLLMService(
                    api_key=self.openai_api_key,
                    model=self.openai_model,
                    session_properties=session_properties,
                    start_audio_paused=False,
                )
                logger.info(f"✅ OpenAI Service created: {type(self.openai_service).__name__}")

                disconnect_tool_handler = create_disconnect_tool_handler(self.websocket_transport)
                self.openai_service.register_function("disconnect_client", disconnect_tool_handler)
                logger.info("✅ Registered disconnect tool handler")

                if self.mcp_client and mcp_tools_schema:
                    try:
                        await self.mcp_client.register_tools_schema(mcp_tools_schema, self.openai_service)
                        logger.info(f"✅ Registered {len(mcp_tools_schema.standard_tools)} MCP tool handlers")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to register MCP tool handlers: {e}")

            self.openai_service._settings.session_properties = session_properties

            if client_id and self.session_manager:
                runtime_aggregator = self.session_manager.activate_runtime_context(client_id, "server")
                self.session_manager.set_current_service(client_id, self.openai_service)
                await self.openai_service.reset_runtime_session(
                    context=runtime_aggregator.user().context,
                    session_properties=session_properties,
                )
                logger.info(f"✅ Refreshed OpenAI runtime session for client {client_id}")

            return self.openai_service
    
    async def run(self) -> None:
        """Run the application."""
        await self.initialize()
        
        # Create initial OpenAI service (will be replaced per connection)
        await self._ensure_openai_service()
        
        # Build pipeline - based on pipecat-examples, one pipeline handles all connections
        # The transport manages multiple connections internally
        self._build_pipeline_for_transport(self.websocket_transport, "server")
        
        # Setup WebSocket event handlers
        async def on_client_connected(client_id: str):
            """Handle new client connection."""
            await self._ensure_openai_service(client_id=client_id)
            if self.audio_recording_service:
                self.audio_recording_service.start_new_session(client_id)
        
        async def on_client_disconnected(client_id: str):
            """Handle client disconnection."""
            if self.session_manager:
                self.session_manager.handle_client_disconnect(client_id, self.openai_service)
            if self.audio_recording_service:
                self.audio_recording_service.stop_recording()
            if self.openai_service:
                await self.openai_service.close_runtime_session()
        
        self.websocket_handler.setup_event_handlers(
            transport=self.websocket_transport,
            on_client_connected_callback=on_client_connected,
            on_client_disconnected_callback=on_client_disconnected,
        )
        
        try:
            # Start the pipeline runner - this will start the WebSocket server
            # Based on pipecat-examples: PipelineRunner.run() starts the transport server
            logger.info("✅ Starting WebSocket server and pipeline...")
            await self.runner.run(self.current_task)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Cleanup resources."""
        logger.info("Gracefully shutting down application...")
        
        if self.runner:
            try:
                await self.runner.cancel()
            except Exception as e:
                logger.warning(f"⚠️ Error cancelling runner: {e}")
        
        if self.websocket_handler:
            try:
                await self.websocket_handler.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up WebSocket handler: {e}")
        
        if self.audio_recording_service:
            self.audio_recording_service.cleanup()
        
        logger.info("✅ Application shutdown complete")


async def main() -> None:
    """Main entry point."""
    app = Application()
    
    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
