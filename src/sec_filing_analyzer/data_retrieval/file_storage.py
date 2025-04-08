"""
File Storage Module

This module provides functionality for saving and loading SEC filings to/from disk.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import shutil
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileStorage:
    """
    Handles saving and loading SEC filings to/from disk.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None
    ):
        """
        Initialize the file storage.

        Args:
            base_dir: Base directory for all filing storage
        """
        # Set up directories
        self.base_dir = base_dir or Path("data/filings")
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.cache_dir = self.base_dir / "cache"
        self.html_dir = self.base_dir / "html"
        self.xml_dir = self.base_dir / "xml"

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(exist_ok=True)
        self.processed_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        self.html_dir.mkdir(exist_ok=True)
        self.xml_dir.mkdir(exist_ok=True)

        logger.info(f"Initialized file storage with base directory: {self.base_dir}")

    def save_raw_filing(
        self,
        filing_id: str,
        content: str,
        metadata: Dict[str, Any],
        format: str = "txt"
    ) -> Path:
        """
        Save raw filing content to disk.

        Args:
            filing_id: Unique identifier for the filing (e.g., accession number)
            content: Raw filing content
            metadata: Filing metadata
            format: File format (txt, html, etc.)

        Returns:
            Path to the saved file
        """
        # Create company directory
        company_dir = self.raw_dir / metadata.get("ticker", "unknown")
        company_dir.mkdir(exist_ok=True)

        # Create year directory
        year = metadata.get("filing_date", "").split("-")[0]
        year_dir = company_dir / year
        year_dir.mkdir(exist_ok=True)

        # Create file path
        file_path = year_dir / f"{filing_id}.{format}"

        # Save content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

        # Save metadata
        metadata_path = year_dir / f"{filing_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved raw filing {filing_id} to {file_path}")
        return file_path

    def save_html_filing(
        self,
        filing_id: str,
        html_content: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save HTML filing content to disk.

        Args:
            filing_id: Unique identifier for the filing
            html_content: HTML filing content
            metadata: Filing metadata

        Returns:
            Path to the saved file
        """
        # Create company directory
        company_dir = self.html_dir / metadata.get("ticker", "unknown")
        company_dir.mkdir(exist_ok=True)

        # Create year directory
        year = metadata.get("filing_date", "").split("-")[0]
        year_dir = company_dir / year
        year_dir.mkdir(exist_ok=True)

        # Create file path
        file_path = year_dir / f"{filing_id}.html"

        # Save HTML content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Save metadata
        metadata_path = year_dir / f"{filing_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved HTML filing {filing_id} to {file_path}")
        return file_path

    def save_xml_filing(
        self,
        filing_id: str,
        xml_content: str,
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save XML filing content to disk.

        Args:
            filing_id: Unique identifier for the filing
            xml_content: XML content
            metadata: Filing metadata

        Returns:
            Path to the saved file
        """
        # Create company directory
        company_dir = self.xml_dir / metadata.get("ticker", "unknown")
        company_dir.mkdir(exist_ok=True)

        # Create year directory
        year = metadata.get("filing_date", "").split("-")[0]
        year_dir = company_dir / year
        year_dir.mkdir(exist_ok=True)

        # Create file path
        file_path = year_dir / f"{filing_id}.xml"

        # Save content
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(xml_content)

        # Save metadata
        metadata_path = year_dir / f"{filing_id}_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Saved XML filing {filing_id} to {file_path}")
        return file_path

    def save_processed_filing(
        self,
        filing_id: str,
        processed_data: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Path:
        """
        Save processed filing data to disk.

        Args:
            filing_id: Unique identifier for the filing
            processed_data: Processed filing data
            metadata: Filing metadata

        Returns:
            Path to the saved file
        """
        # Create company directory
        company_dir = self.processed_dir / metadata.get("ticker", "unknown")
        company_dir.mkdir(exist_ok=True)

        # Create year directory
        year = metadata.get("filing_date", "").split("-")[0]
        year_dir = company_dir / year
        year_dir.mkdir(exist_ok=True)

        # Create file path
        file_path = year_dir / f"{filing_id}_processed.json"

        # Combine data and metadata
        data = {
            "metadata": metadata,
            "processed_data": processed_data
        }

        # Save data
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Saved processed filing {filing_id} to {file_path}")
        return file_path

    def cache_filing(
        self,
        filing_id: str,
        filing_data: Dict[str, Any]
    ) -> Path:
        """
        Cache filing data for quick access.

        Args:
            filing_id: Unique identifier for the filing
            filing_data: Filing data to cache

        Returns:
            Path to the cached file
        """
        # Create file path - use a flat structure for cache
        file_path = self.cache_dir / f"{filing_id}.json"

        # Save data
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(filing_data, f, indent=2, default=str)

        logger.info(f"Cached filing {filing_id} to {file_path}")
        return file_path

    def load_raw_filing(
        self,
        filing_id: str,
        ticker: Optional[str] = None,
        year: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load raw filing from disk.

        Args:
            filing_id: Unique identifier for the filing
            ticker: Optional company ticker symbol
            year: Optional filing year

        Returns:
            Dictionary with content and metadata, or None if not found
        """
        # Try to find the filing if ticker and year are not provided
        if not ticker or not year:
            # Search in all directories
            for company_dir in self.raw_dir.glob("*"):
                if not company_dir.is_dir():
                    continue

                for year_dir in company_dir.glob("*"):
                    if not year_dir.is_dir():
                        continue

                    content_path = year_dir / f"{filing_id}.txt"
                    metadata_path = year_dir / f"{filing_id}_metadata.json"

                    if content_path.exists() and metadata_path.exists():
                        ticker = company_dir.name
                        year = year_dir.name
                        break

                if ticker and year:
                    break

        # If still not found, return None
        if not ticker or not year:
            logger.warning(f"Raw filing {filing_id} not found")
            return None

        # Create file paths
        content_path = self.raw_dir / ticker / year / f"{filing_id}.txt"
        metadata_path = self.raw_dir / ticker / year / f"{filing_id}_metadata.json"

        # Check if files exist
        if not content_path.exists() or not metadata_path.exists():
            logger.warning(f"Raw filing {filing_id} not found")
            return None

        # Load content
        with open(content_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return {
            "content": content,
            "metadata": metadata
        }

    def load_html_filing(
        self,
        filing_id: str,
        ticker: Optional[str] = None,
        year: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load HTML filing from disk.

        Args:
            filing_id: Unique identifier for the filing
            ticker: Optional company ticker symbol
            year: Optional filing year

        Returns:
            Dictionary with HTML content and metadata, or None if not found
        """
        # Try to find the filing if ticker and year are not provided
        if not ticker or not year:
            # Search in all directories
            for company_dir in self.html_dir.glob("*"):
                if not company_dir.is_dir():
                    continue

                for year_dir in company_dir.glob("*"):
                    if not year_dir.is_dir():
                        continue

                    html_path = year_dir / f"{filing_id}.html"
                    metadata_path = year_dir / f"{filing_id}_metadata.json"

                    if html_path.exists() and metadata_path.exists():
                        ticker = company_dir.name
                        year = year_dir.name
                        break

                if ticker and year:
                    break

        # If still not found, return None
        if not ticker or not year:
            logger.warning(f"HTML filing {filing_id} not found")
            return None

        # Create file paths
        html_path = self.html_dir / ticker / year / f"{filing_id}.html"
        metadata_path = self.html_dir / ticker / year / f"{filing_id}_metadata.json"

        # Check if files exist
        if not html_path.exists() or not metadata_path.exists():
            logger.warning(f"HTML filing {filing_id} not found")
            return None

        # Load HTML content
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return {
            "html_content": html_content,
            "metadata": metadata
        }

    def load_xml_filing(
        self,
        filing_id: str,
        ticker: Optional[str] = None,
        year: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load XML filing from disk.

        Args:
            filing_id: Unique identifier for the filing
            ticker: Optional company ticker symbol
            year: Optional filing year

        Returns:
            Dictionary with XML content and metadata, or None if not found
        """
        # Try to find the filing if ticker and year are not provided
        if not ticker or not year:
            # Search in all directories
            for company_dir in self.xml_dir.glob("*"):
                if not company_dir.is_dir():
                    continue

                for year_dir in company_dir.glob("*"):
                    if not year_dir.is_dir():
                        continue

                    xml_path = year_dir / f"{filing_id}.xml"
                    metadata_path = year_dir / f"{filing_id}_metadata.json"

                    if xml_path.exists() and metadata_path.exists():
                        ticker = company_dir.name
                        year = year_dir.name
                        break

                if ticker and year:
                    break

        # If still not found, return None
        if not ticker or not year:
            logger.warning(f"XML filing {filing_id} not found")
            return None

        # Create file paths
        xml_path = self.xml_dir / ticker / year / f"{filing_id}.xml"
        metadata_path = self.xml_dir / ticker / year / f"{filing_id}_metadata.json"

        # Check if files exist
        if not xml_path.exists() or not metadata_path.exists():
            logger.warning(f"XML filing {filing_id} not found")
            return None

        # Load XML content
        with open(xml_path, "r", encoding="utf-8") as f:
            xml_content = f.read()

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return {
            "xml_content": xml_content,
            "metadata": metadata
        }

    def load_processed_filing(
        self,
        filing_id: str,
        ticker: Optional[str] = None,
        year: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Load processed filing from disk.

        Args:
            filing_id: Unique identifier for the filing
            ticker: Optional company ticker symbol
            year: Optional filing year

        Returns:
            Processed filing data, or None if not found
        """
        # Try to find the filing if ticker and year are not provided
        if not ticker or not year:
            # Search in all directories
            for company_dir in self.processed_dir.glob("*"):
                if not company_dir.is_dir():
                    continue

                for year_dir in company_dir.glob("*"):
                    if not year_dir.is_dir():
                        continue

                    file_path = year_dir / f"{filing_id}_processed.json"

                    if file_path.exists():
                        ticker = company_dir.name
                        year = year_dir.name
                        break

                if ticker and year:
                    break

        # If still not found, return None
        if not ticker or not year:
            logger.warning(f"Processed filing {filing_id} not found")
            return None

        # Create file path
        file_path = self.processed_dir / ticker / year / f"{filing_id}_processed.json"

        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Processed filing {filing_id} not found")
            return None

        # Load data
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def load_cached_filing(
        self,
        filing_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Load cached filing from disk.

        Args:
            filing_id: Unique identifier for the filing

        Returns:
            Cached filing data, or None if not found
        """
        # Create file path - using flat structure for cache
        file_path = self.cache_dir / f"{filing_id}.json"

        # Check if file exists
        if not file_path.exists():
            logger.warning(f"Cached filing {filing_id} not found")
            return None

        # Load data
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data

    def list_filings(
        self,
        ticker: Optional[str] = None,
        year: Optional[str] = None,
        filing_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available filings.

        Args:
            ticker: Optional company ticker symbol
            year: Optional filing year
            filing_type: Optional filing type

        Returns:
            List of filing metadata
        """
        filings = []

        # Determine search path
        if ticker and year:
            search_path = self.raw_dir / ticker / year
        elif ticker:
            search_path = self.raw_dir / ticker
        else:
            search_path = self.raw_dir

        # Check if path exists
        if not search_path.exists():
            logger.warning(f"Search path {search_path} not found")
            return filings

        # Find metadata files
        for metadata_path in search_path.glob("**/*_metadata.json"):
            # Load metadata
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)

            # Apply filters
            if filing_type and metadata.get("form") != filing_type:
                continue

            filings.append(metadata)

        return filings

    def export_filing(
        self,
        filing_id: str,
        ticker: Optional[str] = None,
        year: Optional[str] = None,
        export_dir: Optional[Path] = None
    ) -> Path:
        """
        Export a filing to a specified directory.

        Args:
            filing_id: Unique identifier for the filing
            ticker: Optional company ticker symbol
            year: Optional filing year
            export_dir: Directory to export to

        Returns:
            Path to the exported directory
        """
        # Find the filing if ticker and year are not provided
        if not ticker or not year:
            # Search in all directories
            for company_dir in self.raw_dir.glob("*"):
                if not company_dir.is_dir():
                    continue

                for year_dir in company_dir.glob("*"):
                    if not year_dir.is_dir():
                        continue

                    content_path = year_dir / f"{filing_id}.txt"
                    metadata_path = year_dir / f"{filing_id}_metadata.json"

                    if content_path.exists() and metadata_path.exists():
                        ticker = company_dir.name
                        year = year_dir.name
                        break

                if ticker and year:
                    break

        # If still not found, raise an error
        if not ticker or not year:
            raise ValueError(f"Filing {filing_id} not found")

        # Create export directory
        export_dir = export_dir or Path("exports") / ticker / year / filing_id
        export_dir.mkdir(parents=True, exist_ok=True)

        # Copy raw filing
        raw_content_path = self.raw_dir / ticker / year / f"{filing_id}.txt"
        raw_metadata_path = self.raw_dir / ticker / year / f"{filing_id}_metadata.json"

        if raw_content_path.exists():
            shutil.copy2(raw_content_path, export_dir / "raw.txt")

        if raw_metadata_path.exists():
            shutil.copy2(raw_metadata_path, export_dir / "raw_metadata.json")

        # Copy HTML filing if available
        html_path = self.html_dir / ticker / year / f"{filing_id}.html"
        html_metadata_path = self.html_dir / ticker / year / f"{filing_id}_metadata.json"

        if html_path.exists():
            shutil.copy2(html_path, export_dir / "filing.html")

        if html_metadata_path.exists():
            shutil.copy2(html_metadata_path, export_dir / "html_metadata.json")

        # Copy XML filing if available
        xml_path = self.xml_dir / ticker / year / f"{filing_id}.xml"
        xml_metadata_path = self.xml_dir / ticker / year / f"{filing_id}_metadata.json"

        if xml_path.exists():
            shutil.copy2(xml_path, export_dir / "filing.xml")

        if xml_metadata_path.exists():
            shutil.copy2(xml_metadata_path, export_dir / "xml_metadata.json")

        # Copy processed filing
        processed_path = self.processed_dir / ticker / year / f"{filing_id}_processed.json"

        if processed_path.exists():
            shutil.copy2(processed_path, export_dir / "processed.json")

        logger.info(f"Exported filing {filing_id} to {export_dir}")
        return export_dir