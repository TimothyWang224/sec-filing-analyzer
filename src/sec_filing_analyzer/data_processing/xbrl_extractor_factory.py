"""
Factory for creating XBRL extractors.

This module provides a factory class for creating XBRL extractors based on
configuration or availability.
"""

import logging
from typing import Optional, Union

from .simplified_xbrl_extractor import SimplifiedXBRLExtractor
from .edgar_xbrl_extractor import EdgarXBRLExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XBRLExtractorFactory:
    """
    Factory for creating XBRL extractors.
    
    This class provides methods for creating XBRL extractors based on
    configuration or availability.
    """
    
    @staticmethod
    def create_extractor(extractor_type: str = "edgar", cache_dir: Optional[str] = None) -> Union[SimplifiedXBRLExtractor, EdgarXBRLExtractor]:
        """
        Create an XBRL extractor based on the specified type.
        
        Args:
            extractor_type: Type of extractor to create ("edgar" or "simplified")
            cache_dir: Optional directory to cache extracted data
            
        Returns:
            An XBRL extractor instance
        """
        if extractor_type.lower() == "edgar":
            try:
                # Try to import edgar to check if it's available
                import edgar
                logger.info("Using EdgarXBRLExtractor")
                return EdgarXBRLExtractor(cache_dir=cache_dir)
            except ImportError:
                logger.warning("Edgar library not available, falling back to SimplifiedXBRLExtractor")
                return SimplifiedXBRLExtractor(cache_dir=cache_dir)
        else:
            logger.info("Using SimplifiedXBRLExtractor")
            return SimplifiedXBRLExtractor(cache_dir=cache_dir)
    
    @staticmethod
    def get_default_extractor(cache_dir: Optional[str] = None) -> Union[SimplifiedXBRLExtractor, EdgarXBRLExtractor]:
        """
        Get the default XBRL extractor.
        
        This method tries to use the EdgarXBRLExtractor if the edgar library is available,
        otherwise it falls back to the SimplifiedXBRLExtractor.
        
        Args:
            cache_dir: Optional directory to cache extracted data
            
        Returns:
            An XBRL extractor instance
        """
        try:
            # Try to import edgar to check if it's available
            import edgar
            logger.info("Using EdgarXBRLExtractor as default")
            return EdgarXBRLExtractor(cache_dir=cache_dir)
        except ImportError:
            logger.warning("Edgar library not available, using SimplifiedXBRLExtractor as default")
            return SimplifiedXBRLExtractor(cache_dir=cache_dir)
