"""Whisper STT service for speech recognition."""

import io
import tempfile
from typing import Optional, Dict, List
import asyncio
import os

from ..core.config import settings
from ..core.logging import get_logger

logger = get_logger(__name__)


class WhisperSTTService:
    """Whisper-based Speech-to-Text service."""
    
    def __init__(self):
        self.model = None
        self._model_loaded = False
    
    def _load_model(self):
        """Load Whisper model on first use."""
        if self._model_loaded:
            return
        
        try:
            import faster_whisper
            
            # Use tiny model by default for faster loading and lower resource usage
            model_size = getattr(settings, 'whisper_model', 'tiny')
            compute_type = getattr(settings, 'whisper_compute', 'cpu')
            
            logger.info(f"Loading Whisper model: {model_size} on {compute_type}")
            
            # Try to load the model with error handling
            self.model = faster_whisper.WhisperModel(
                model_size, 
                device=compute_type,
                compute_type="int8" if compute_type == "cpu" else "float16",
                download_root=None,  # Use default cache directory
                local_files_only=False  # Allow downloads if needed
            )
            
            self._model_loaded = True
            logger.info(f"Whisper model '{model_size}' loaded successfully")
            
        except ImportError as e:
            logger.error(f"faster-whisper not installed: {e}")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.model = None
            
            # Try with tiny model as fallback
            if model_size != 'tiny':
                try:
                    logger.info("Trying fallback to tiny model...")
                    import faster_whisper
                    self.model = faster_whisper.WhisperModel(
                        'tiny', 
                        device='cpu',
                        compute_type="int8"
                    )
                    self._model_loaded = True
                    logger.info("Fallback to tiny model successful")
                except Exception as fallback_error:
                    logger.error(f"Fallback to tiny model also failed: {fallback_error}")
                    self.model = None
    
    async def transcribe_audio(
        self, 
        audio_data: bytes, 
        language: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio file bytes
            language: Optional language hint (e.g., 'en', 'es')
            
        Returns:
            Dictionary with transcribed text and detected language
        """
        if not self.model:
            self._load_model()
        
        if not self.model:
            return {"text": "", "language": "en", "error": "Whisper model not available"}
        
        try:
            # Write audio data to temporary file
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file_path = temp_file.name
            
            logger.info(f"Processing audio file: {temp_file_path} ({len(audio_data)} bytes)")
            
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._transcribe_sync, 
                temp_file_path, 
                language
            )
            
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup temp file: {cleanup_error}")
            
            return result
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {"text": "", "language": language or "en", "error": str(e)}
    
    def _transcribe_sync(self, audio_path: str, language: Optional[str] = None) -> Dict[str, str]:
        """Synchronous transcription method."""
        try:
            # Configure transcription parameters - make them more lenient
            beam_size = getattr(settings, 'whisper_beam_size', 1)
            temperature = getattr(settings, 'whisper_temperature', 0.0)
            no_speech_threshold = getattr(settings, 'whisper_no_speech_threshold', 0.2)  # Lower threshold
            
            logger.info(f"Starting Whisper transcription with language hint: {language}")
            logger.info(f"Audio file size: {os.path.getsize(audio_path)} bytes")
            
            # Transcribe with safer decoding settings to avoid repetitions
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=max(beam_size, 3),
                temperature=temperature,
                no_speech_threshold=no_speech_threshold,
                vad_filter=True,  # Use VAD to trim silences that can cause repeats
                word_timestamps=False,
                condition_on_previous_text=True
            )
            
            # Combine all segments
            text_parts = []
            last_clean = ""
            segment_count = 0
            for segment in segments:
                segment_count += 1
                logger.info(f"Segment {segment_count}: '{segment.text}' (confidence: {getattr(segment, 'avg_logprob', 'N/A')})")
                if segment.text.strip():
                    current = segment.text.strip()
                    # Drop exact duplicates or short loops like "How are you? How are you?"
                    if current == last_clean:
                        continue
                    # Simple dedupe of repeated bigrams within the segment
                    words = current.split()
                    if len(words) > 6:
                        deduped_words = []
                        seen_bigrams = set()
                        for i, w in enumerate(words):
                            deduped_words.append(w)
                            if i > 0:
                                bigram = (words[i-1].lower(), w.lower())
                                if bigram in seen_bigrams:
                                    # truncate on repeat loop
                                    current = " ".join(deduped_words)
                                    break
                                seen_bigrams.add(bigram)
                    text_parts.append(current)
                    last_clean = current
            
            text = " ".join(text_parts)
            
            detected_language = info.language if hasattr(info, 'language') else (language or "en")
            confidence = info.language_probability if hasattr(info, 'language_probability') else 0.0
            
            logger.info(f"Whisper transcription result: segments={segment_count}, text='{text}', lang={detected_language}, conf={confidence:.2f}")
            
            if not text.strip():
                # Try once more with even more lenient settings
                logger.info("No text found, trying with maximum lenient settings...")
                segments, info = self.model.transcribe(
                    audio_path,
                    language=None,  # Auto-detect
                    beam_size=3,
                    temperature=0.2,  # Lower temp to avoid hallucinations
                    no_speech_threshold=0.1,  # Very low threshold
                    vad_filter=True
                )
                
                text_parts = []
                last_clean = ""
                for segment in segments:
                    logger.info(f"Lenient segment: '{segment.text}'")
                    if segment.text.strip():
                        current = segment.text.strip()
                        if current == last_clean:
                            continue
                        text_parts.append(current)
                        last_clean = current
                
                text = " ".join(text_parts)
                
                if not text.strip():
                    return {"text": "", "language": detected_language, "error": "No speech detected in audio"}
            
            return {
                "text": text.strip(),
                "language": detected_language,
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Whisper sync transcription failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"text": "", "language": language or "en", "error": str(e)}
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of languages supported by Whisper."""
        return [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "tr", "name": "Turkish"},
            {"code": "pl", "name": "Polish"},
            {"code": "nl", "name": "Dutch"},
            {"code": "sv", "name": "Swedish"},
            {"code": "da", "name": "Danish"},
            {"code": "no", "name": "Norwegian"},
            {"code": "fi", "name": "Finnish"},
            {"code": "uk", "name": "Ukrainian"},
            {"code": "he", "name": "Hebrew"},
            {"code": "th", "name": "Thai"},
            {"code": "vi", "name": "Vietnamese"},
            {"code": "id", "name": "Indonesian"},
            {"code": "ms", "name": "Malay"},
            {"code": "cs", "name": "Czech"},
            {"code": "sk", "name": "Slovak"},
            {"code": "hu", "name": "Hungarian"},
            {"code": "ro", "name": "Romanian"},
            {"code": "bg", "name": "Bulgarian"},
            {"code": "hr", "name": "Croatian"},
            {"code": "sr", "name": "Serbian"},
            {"code": "sl", "name": "Slovenian"},
            {"code": "et", "name": "Estonian"},
            {"code": "lv", "name": "Latvian"},
            {"code": "lt", "name": "Lithuanian"},
            {"code": "fa", "name": "Persian"},
            {"code": "ur", "name": "Urdu"},
            {"code": "bn", "name": "Bengali"},
            {"code": "ta", "name": "Tamil"},
            {"code": "te", "name": "Telugu"},
            {"code": "ml", "name": "Malayalam"},
            {"code": "kn", "name": "Kannada"},
            {"code": "gu", "name": "Gujarati"},
            {"code": "pa", "name": "Punjabi"},
            {"code": "ne", "name": "Nepali"},
            {"code": "si", "name": "Sinhala"},
            {"code": "my", "name": "Myanmar"},
            {"code": "km", "name": "Khmer"},
            {"code": "lo", "name": "Lao"},
            {"code": "ka", "name": "Georgian"},
            {"code": "am", "name": "Amharic"},
            {"code": "sw", "name": "Swahili"},
            {"code": "yo", "name": "Yoruba"},
            {"code": "zu", "name": "Zulu"},
            {"code": "af", "name": "Afrikaans"},
            {"code": "sq", "name": "Albanian"},
            {"code": "az", "name": "Azerbaijani"},
            {"code": "be", "name": "Belarusian"},
            {"code": "bs", "name": "Bosnian"},
            {"code": "ca", "name": "Catalan"},
            {"code": "cy", "name": "Welsh"},
            {"code": "eu", "name": "Basque"},
            {"code": "gl", "name": "Galician"},
            {"code": "is", "name": "Icelandic"},
            {"code": "ga", "name": "Irish"},
            {"code": "mk", "name": "Macedonian"},
            {"code": "mt", "name": "Maltese"},
            {"code": "mn", "name": "Mongolian"},
            {"code": "lb", "name": "Luxembourgish"},
            {"code": "mi", "name": "Maori"},
            {"code": "oc", "name": "Occitan"},
            {"code": "br", "name": "Breton"},
            {"code": "fo", "name": "Faroese"},
            {"code": "hy", "name": "Armenian"},
            {"code": "kk", "name": "Kazakh"},
            {"code": "ky", "name": "Kyrgyz"},
            {"code": "tg", "name": "Tajik"},
            {"code": "tk", "name": "Turkmen"},
            {"code": "uz", "name": "Uzbek"},
            {"code": "tt", "name": "Tatar"},
            {"code": "ba", "name": "Bashkir"},
            {"code": "cv", "name": "Chuvash"},
            {"code": "sah", "name": "Yakut"},
            {"code": "yue", "name": "Cantonese"},
            {"code": "nn", "name": "Nynorsk"},
            {"code": "haw", "name": "Hawaiian"},
            {"code": "ln", "name": "Lingala"},
            {"code": "mg", "name": "Malagasy"},
            {"code": "bo", "name": "Tibetan"},
            {"code": "tl", "name": "Tagalog"},
            {"code": "jw", "name": "Javanese"},
            {"code": "su", "name": "Sundanese"}
        ]


# Global service instance
whisper_stt_service = WhisperSTTService() 