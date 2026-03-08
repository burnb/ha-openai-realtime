"""Simple serializer for raw binary PCM audio frames."""
import logging
from pipecat.frames.frames import InputAudioRawFrame, OutputAudioRawFrame, Frame
from pipecat.serializers.base_serializer import FrameSerializer

logger = logging.getLogger(__name__)


class RawAudioSerializer(FrameSerializer):
    """Serializer that treats all binary messages as raw PCM audio."""

    async def deserialize(self, message: str | bytes) -> InputAudioRawFrame | None:
        """Deserialize binary message as raw PCM audio frame.
        
        Args:
            message: Binary PCM audio data (16-bit, 24kHz, mono)
            
        Returns:
            InputAudioRawFrame with the audio data, or None if invalid
        """
        if not isinstance(message, bytes):
            # Skip non-binary messages (text/JSON)
            return None
            
        # Validate audio format: 16-bit = 2 bytes per sample
        if len(message) % 2 != 0:
            logger.warning(f"⚠️ Received audio with odd byte count: {len(message)} bytes, skipping")
            return None
        
        # Create InputAudioRawFrame
        # Audio is 24kHz, 16-bit, mono PCM
        frame = InputAudioRawFrame(
            audio=message,
            sample_rate=24000,
            num_channels=1
        )
        
        return frame
    
    async def serialize(self, frame: Frame) -> bytes | None:
        """Serialize frame to binary message.
        
        For output audio frames, we just return the raw audio bytes.
        Other frames are not serialized.
        """
        if isinstance(frame, OutputAudioRawFrame):
            audio_bytes = frame.audio
            logger.debug(f"📤 Serializing OutputAudioRawFrame: {len(audio_bytes)} bytes")
            return audio_bytes
        # Let the transport skip frames that are not raw output audio.
        logger.debug(f"📤 Skipping non-audio frame during serialization: {type(frame).__name__}")
        return None

