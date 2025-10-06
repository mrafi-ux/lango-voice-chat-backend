"""Mock STT service for testing when Whisper is not available."""

import asyncio
from typing import Optional, Dict, List
import random

from ..core.logging import get_logger

logger = get_logger(__name__)


class MockSTTService:
    """Mock Speech-to-Text service for testing."""
    
    def __init__(self):
        self.model = "mock"
        self._model_loaded = True
        
        # Mock transcriptions for different languages
        self.mock_transcriptions = {
            'en': [
                "Hello, how are you today?",
                "I'm feeling much better now, thank you.",
                "Could you please help me with this?",
                "The weather is nice today.",
                "I need to see the doctor."
            ],
            'es': [
                "Hola, ¿cómo estás hoy?",
                "Me siento mucho mejor ahora, gracias.",
                "¿Podrías ayudarme con esto por favor?",
                "El clima está agradable hoy.",
                "Necesito ver al médico."
            ],
            'fr': [
                "Bonjour, comment allez-vous aujourd'hui?",
                "Je me sens beaucoup mieux maintenant, merci.",
                "Pourriez-vous m'aider avec ceci s'il vous plaît?",
                "Le temps est agréable aujourd'hui.",
                "J'ai besoin de voir le médecin."
            ]
        }
    
    async def transcribe_audio(
        self, 
        audio_data: bytes, 
        language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Mock transcribe audio data to text.
        
        Args:
            audio_data: Audio file bytes (ignored in mock)
            language: Optional language hint (e.g., 'en', 'es')
            
        Returns:
            Dictionary with mock transcribed text and language
        """
        try:
            # Simulate processing time
            await asyncio.sleep(0.5)
            
            # Determine language
            detected_lang = language or 'en'
            if detected_lang not in self.mock_transcriptions:
                detected_lang = 'en'
            
            # Get random mock transcription
            mock_text = random.choice(self.mock_transcriptions[detected_lang])
            
            logger.info(f"Mock STT transcription: '{mock_text}' (lang: {detected_lang})")
            
            return {
                "text": mock_text,
                "language": detected_lang,
                "confidence": 0.95
            }
            
        except Exception as e:
            logger.error(f"Mock transcription failed: {e}")
            return {"text": "", "language": language or "en", "error": str(e)}
    
    def get_supported_languages(self) -> List[str]:
        """Get list of mock supported languages."""
        return list(self.mock_transcriptions.keys())


# Global mock service instance
mock_stt_service = MockSTTService() 