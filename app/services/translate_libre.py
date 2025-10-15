"""Translation service using deep-translator with Google Translate."""

import asyncio
from typing import Optional
from deep_translator import GoogleTranslator

from ..core.logging import get_logger

logger = get_logger(__name__)


class TranslationService:
    """Translation service using deep-translator with Google Translate."""
    
    def __init__(self):
        """Initialize the translation service."""
        self.translator = GoogleTranslator()
        logger.info("Translation service initialized with Google Translate (free)")
    
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text from source language to target language.
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'en', 'es')
            target_lang: Target language code (e.g., 'en', 'es')
            
        Returns:
            Translated text
        """
        if source_lang == target_lang:
            return text
            
        # Normalize language codes (allow 'auto' to trigger autodetect)
        source_lang = (source_lang or "").strip().lower()
        target_lang = (target_lang or "").strip().lower()
        source_code = self._normalize_lang_code(source_lang) if source_lang and source_lang != "auto" else "auto"
        target_code = self._normalize_lang_code(target_lang)
        
        logger.info(f"Translating: '{text[:50]}...' ({source_code} → {target_code})")
        
        try:
            # Run translation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._translate_sync, 
                text, 
                source_code, 
                target_code
            )
            
            # If translation came back identical or empty, retry with autodetect
            if not result or result.strip() == text.strip():
                if source_code != "auto":
                    logger.info("Translation unchanged/empty; retrying with source=auto")
                    result = await loop.run_in_executor(
                        None,
                        self._translate_sync,
                        text,
                        "auto",
                        target_code
                    )
            if result and result.strip():
                logger.info(f"Translation successful: '{result[:50]}...'")
                return result
            else:
                logger.warning("Translation returned empty result")
                return self._fallback_translate(text, source_code, target_code)
                
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return self._fallback_translate(text, source_code, target_code)
    
    def _translate_sync(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Synchronous translation using deep-translator."""
        try:
            # Create translator for specific language pair
            translator = GoogleTranslator(source=source_lang if source_lang else "auto", target=target_lang)
            result = translator.translate(text)
            return result if result else None
        except Exception as e:
            logger.error(f"Google Translate API error: {e}")
            return None
    
    def _normalize_lang_code(self, lang_code: str) -> str:
        """Normalize language code to 2-letter format."""
        # Convert BCP-47 to simple 2-letter codes
        lang_map = {
            "en-US": "en",
            "en-GB": "en", 
            "es-ES": "es",
            "es-MX": "es",
            "fr-FR": "fr",
            "de-DE": "de",
            "it-IT": "it",
            "pt-BR": "pt",
            "pt-PT": "pt",
            "ru-RU": "ru",
            "zh-CN": "zh",
            "ja-JP": "ja",
            "ko-KR": "ko",
            "ar-SA": "ar"
        }
        
        # Return mapped code or extract base language
        base_lang = lang_map.get(lang_code, lang_code.split("-")[0])
        return base_lang.lower()
    
    def _fallback_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Fallback translation when real translation fails."""
        if target_lang == "es":
            return f"[Traducido] {text}"
        elif target_lang == "en":
            return f"[Translated] {text}"
        else:
            return f"[{target_lang.upper()}] {text}"
    
    async def get_supported_languages(self):
        """Get list of supported language pairs."""
        return [
            {"code": "en", "name": "English"},
            {"code": "es", "name": "Spanish"},
            {"code": "fr", "name": "French"},
            {"code": "de", "name": "German"},
            {"code": "it", "name": "Italian"},
            {"code": "pt", "name": "Portuguese"},
            {"code": "ru", "name": "Russian"},
            {"code": "zh", "name": "Chinese"},
            {"code": "ja", "name": "Japanese"},
            {"code": "ko", "name": "Korean"},
            {"code": "ar", "name": "Arabic"},
            {"code": "hi", "name": "Hindi"},
            {"code": "tr", "name": "Turkish"},
            {"code": "pl", "name": "Polish"},
            {"code": "nl", "name": "Dutch"}
        ]


# Global service instance  
translate_service = TranslationService()
