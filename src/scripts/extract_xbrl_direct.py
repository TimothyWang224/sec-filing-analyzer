"""
Direct XBRL extraction script using the edgar library.
"""

import json
import logging
import os
from pathlib import Path

import edgar
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_xbrl_direct():
    """Extract XBRL data directly using the edgar library."""
    try:
        # Create output directory
        output_dir = Path("data/test_output")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set edgar identity from environment variables
        edgar_identity = os.getenv("EDGAR_IDENTITY")
        if edgar_identity:
            edgar.set_identity(edgar_identity)
            logger.info(f"Set edgar identity to: {edgar_identity}")

        # Get Microsoft entity
        logger.info("Getting Microsoft entity...")
        msft = edgar.get_entity("MSFT")
        logger.info(f"Found Microsoft entity with CIK: {msft.cik}")

        # Get a specific filing
        logger.info("Getting a specific filing...")
        accession_number = "0001564590-22-026876"  # Microsoft's 10-K from July 2022

        # Get all filings
        filings = msft.get_filings()
        logger.info(f"Retrieved {len(filings)} filings")

        # Find the filing with the matching accession number
        filing = None
        for f in filings:
            if f.accession_number == accession_number:
                filing = f
                break

        if not filing:
            logger.error(f"Filing with accession number {accession_number} not found")
            return None

        logger.info(f"Found filing: {filing.form} filed on {filing.filing_date}")
        logger.info(f"Filing URL: {filing.filing_url}")

        # Check if the filing has XBRL data
        logger.info("Checking if filing has XBRL data...")
        has_xbrl = hasattr(filing, "is_xbrl") and filing.is_xbrl
        logger.info(f"Filing has XBRL data: {has_xbrl}")

        if has_xbrl:
            # Get XBRL data
            logger.info("Getting XBRL data...")
            xbrl_data = filing.xbrl()

            # Extract basic metadata
            metadata = {
                "filing_id": f"MSFT_{filing.form}_{filing.filing_date.year}",
                "ticker": "MSFT",
                "accession_number": accession_number,
                "filing_date": str(filing.filing_date),
                "filing_type": filing.form,
                "filing_url": filing.filing_url,
            }

            # Extract company information
            if hasattr(xbrl_data, "company"):
                metadata["company"] = xbrl_data.company

            # Extract facts
            facts = []
            if hasattr(xbrl_data, "instance") and hasattr(xbrl_data.instance, "facts"):
                logger.info("Extracting facts...")
                # Get all facts
                all_facts = xbrl_data.instance.facts

                # Process facts
                for index, row in all_facts.iterrows():
                    # Get the concept name
                    concept = index[0] if isinstance(index, tuple) else index

                    # Skip non-US-GAAP facts
                    if not concept.startswith("us-gaap:"):
                        continue

                    # Process each period
                    for period, period_values in row.items():
                        # Skip if no values or not a dict
                        if period_values is None or pd.isna(period_values) or not isinstance(period_values, dict):
                            continue

                        # Process base values (non-dimensional)
                        base_value = period_values.get((), {})
                        if base_value:
                            value = base_value.get("value")
                            if value:
                                # Try to convert to number
                                try:
                                    numeric_value = float(value)
                                except (ValueError, TypeError):
                                    numeric_value = None

                                # Create fact entry
                                fact = {
                                    "concept": concept,
                                    "value": numeric_value if numeric_value is not None else value,
                                    "units": base_value.get("units"),
                                    "decimals": base_value.get("decimals"),
                                    "period": period,
                                    "duration": base_value.get("duration"),
                                }

                                # Parse period information
                                if " to " in period:
                                    start_date, end_date = period.split(" to ")
                                    fact["start_date"] = start_date
                                    fact["end_date"] = end_date
                                    fact["period_type"] = "duration"
                                else:
                                    fact["instant"] = period
                                    fact["period_type"] = "instant"

                                facts.append(fact)

            logger.info(f"Extracted {len(facts)} facts")

            # Extract statements
            statements = {}
            if hasattr(xbrl_data, "statements_dict"):
                logger.info("Extracting statements...")
                # Get all statement keys
                statement_keys = list(xbrl_data.statements_dict.keys())
                logger.info(f"Found {len(statement_keys)} statement keys")

                # Process each statement
                for key in statement_keys:
                    if key.startswith("Statement"):
                        try:
                            # Get the statement
                            statement = xbrl_data.get_statement(key)
                            if statement:
                                # Get statement name
                                statement_name = key.replace("Statement", "").lower()

                                # Map to standard statement names
                                if "income" in statement_name or "operations" in statement_name:
                                    statement_type = "income_statement"
                                elif "balance" in statement_name or "financial position" in statement_name:
                                    statement_type = "balance_sheet"
                                elif "cash" in statement_name:
                                    statement_type = "cash_flow"
                                elif "equity" in statement_name or "stockholder" in statement_name:
                                    statement_type = "equity"
                                elif "comprehensive" in statement_name:
                                    statement_type = "comprehensive_income"
                                else:
                                    statement_type = statement_name

                                # Extract statement metadata
                                statement_data = {
                                    "name": statement.name if hasattr(statement, "name") else key,
                                    "label": statement.label if hasattr(statement, "label") else key,
                                    "periods": statement.periods if hasattr(statement, "periods") else [],
                                }

                                # Extract line items
                                if hasattr(statement, "line_items"):
                                    line_items = []
                                    for item in statement.line_items:
                                        line_item = {
                                            "concept": item.concept if hasattr(item, "concept") else None,
                                            "label": item.label if hasattr(item, "label") else None,
                                            "level": item.level if hasattr(item, "level") else 0,
                                        }

                                        # Extract values
                                        if hasattr(item, "values"):
                                            values = {}
                                            for (
                                                period,
                                                period_values,
                                            ) in item.values.items():
                                                period_dict = {}
                                                for (
                                                    dim_key,
                                                    dim_value,
                                                ) in period_values.items():
                                                    if dim_key == ():
                                                        # Base value
                                                        period_dict["value"] = dim_value.get("value")
                                                    else:
                                                        # Skip dimensional values for simplicity
                                                        pass
                                                values[period] = period_dict
                                            line_item["values"] = values

                                        line_items.append(line_item)

                                    statement_data["line_items"] = line_items

                                statements[statement_type] = statement_data
                        except Exception as e:
                            logger.error(f"Error extracting statement {key}: {e}")

            logger.info(f"Extracted {len(statements)} statements")

            # Create the final data structure
            xbrl_result = {
                "metadata": metadata,
                "facts": facts,
                "statements": statements,
            }

            # Save to file
            output_file = output_dir / "msft_xbrl_direct.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(xbrl_result, f, indent=2, default=str)

            logger.info(f"Saved XBRL data to {output_file}")

            return xbrl_result
        else:
            logger.warning("Filing does not have XBRL data")
            return None

    except Exception as e:
        logger.error(f"Error in extraction: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    extract_xbrl_direct()
