"""
Optimized parallel processor for SEC filings.

This module provides an optimized parallel processor for SEC filings
that balances throughput with API rate limits.
"""

import concurrent.futures
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import ConfigProvider, ETLConfig
from ..data_retrieval.parallel_filing_processor import ParallelFilingProcessor
from ..data_retrieval.sec_downloader import SECFilingsDownloader
from ..semantic.embeddings.robust_embedding_generator import RobustEmbeddingGenerator
from ..storage.graph_store import GraphStore
from ..storage.vector_store import LlamaIndexVectorStore
from ..utils.adaptive_rate_limiter import AdaptiveRateLimiter
from ..utils.etl_logging import (
    generate_run_id,
    log_api_call,
    log_company_processing,
    log_embedding_stats,
    log_etl_end,
    log_etl_start,
    log_filing_processing,
    log_phase_timing,
    log_rate_limit_adjustment,
    setup_etl_logging,
)

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for the processing run."""

    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_filings: int = 0
    processed_filings: int = 0
    failed_filings: int = 0
    skipped_filings: int = 0
    embedding_stats: Dict[str, Any] = field(default_factory=dict)

    def calculate_duration(self) -> float:
        """Calculate the duration of the processing run."""
        if self.end_time is None:
            self.end_time = time.time()
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to a dictionary."""
        duration = self.calculate_duration()
        return {
            "duration_seconds": duration,
            "duration_formatted": f"{duration:.2f}s",
            "total_filings": self.total_filings,
            "processed_filings": self.processed_filings,
            "failed_filings": self.failed_filings,
            "skipped_filings": self.skipped_filings,
            "success_rate": f"{(self.processed_filings / max(1, self.total_filings)) * 100:.1f}%",
            "processing_rate": f"{self.processed_filings / max(1, duration):.2f} filings/second",
            "embedding_stats": self.embedding_stats,
        }


class OptimizedParallelProcessor:
    """
    Optimized parallel processor for SEC filings.

    Features:
    - Adaptive rate limiting based on API response patterns
    - Dynamic worker allocation based on filing complexity
    - Efficient batching of embedding requests
    - Comprehensive error handling and recovery
    - Detailed logging and performance metrics
    """

    def __init__(
        self,
        max_workers: int = 4,
        initial_rate_limit: float = 0.5,
        batch_size: int = 50,
        vector_store: Optional[Any] = None,
        graph_store: Optional[GraphStore] = None,
        file_storage: Optional[Any] = None,
        sec_downloader: Optional[SECFilingsDownloader] = None,
        process_semantic: bool = True,
        process_quantitative: bool = True,
    ):
        """Initialize the optimized parallel processor.

        Args:
            max_workers: Maximum number of worker threads
            initial_rate_limit: Initial rate limit in seconds
            batch_size: Batch size for embedding generation
            vector_store: Vector store for semantic data
            graph_store: Graph store for relationship data
            file_storage: File storage for raw filings
            sec_downloader: SEC filings downloader
            process_semantic: Whether to process semantic data
            process_quantitative: Whether to process quantitative data
        """
        # Initialize components
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.vector_store = vector_store or LlamaIndexVectorStore()
        self.graph_store = graph_store or GraphStore()
        self.file_storage = file_storage
        self.sec_downloader = sec_downloader or SECFilingsDownloader()
        self.process_semantic = process_semantic
        self.process_quantitative = process_quantitative

        # Create shared state for rate limiter
        self.rate_limiter_state = {
            "rate_limit": initial_rate_limit,
            "last_request_time": 0,
            "consecutive_successes": 0,
            "consecutive_failures": 0,
            "total_requests": 0,
            "total_successes": 0,
            "total_failures": 0,
            "lock": threading.Lock(),
        }

        # Create adaptive rate limiter
        self.rate_limiter = AdaptiveRateLimiter(
            initial_rate_limit=initial_rate_limit, shared_state=self.rate_limiter_state
        )

        # Create robust embedding generator with adaptive rate limiting
        self.embedding_generator = RobustEmbeddingGenerator(
            model=ConfigProvider.get_config(ETLConfig).embedding_model,
            max_tokens_per_chunk=8000,  # Safe limit below the 8192 max
            rate_limit=initial_rate_limit,
            batch_size=batch_size,
            max_retries=5,
        )

        # Create filing processor
        self.filing_processor = ParallelFilingProcessor(
            vector_store=self.vector_store,
            graph_store=self.graph_store,
            file_storage=self.file_storage,
            max_workers=max_workers,
        )

        # Initialize stats
        self.stats = ProcessingStats()

        # Set up ETL logging
        setup_etl_logging()

        # Initialize run_id
        self.run_id = None

    def _estimate_filing_complexity(self, filing: Dict[str, Any]) -> int:
        """Estimate the complexity of a filing based on its metadata.

        Args:
            filing: Filing metadata

        Returns:
            Complexity score (higher means more complex)
        """
        # Start with base complexity
        complexity = 1

        # Add complexity based on filing type
        filing_type = filing.get("type", "")
        if filing_type == "10-K":
            complexity += 5  # 10-Ks are very complex
        elif filing_type == "10-Q":
            complexity += 3  # 10-Qs are moderately complex
        elif filing_type == "8-K":
            complexity += 1  # 8-Ks are simpler

        # Add complexity based on file size if available
        file_size = filing.get("file_size", 0)
        if file_size > 1000000:  # > 1MB
            complexity += 3
        elif file_size > 500000:  # > 500KB
            complexity += 2
        elif file_size > 100000:  # > 100KB
            complexity += 1

        return complexity

    def _allocate_workers(self, filings: List[Dict[str, Any]]) -> int:
        """Allocate an appropriate number of workers based on filings.

        Args:
            filings: List of filings to process

        Returns:
            Number of workers to use
        """
        if not filings:
            return 1

        # Calculate total complexity
        total_complexity = sum(
            self._estimate_filing_complexity(filing) for filing in filings
        )

        # Allocate workers based on complexity and filing count
        filing_count = len(filings)

        if filing_count == 1:
            # For a single filing, use fewer workers
            return min(2, self.max_workers)
        elif filing_count <= 3:
            # For a few filings, use moderate parallelism
            return min(filing_count, max(2, self.max_workers // 2))
        else:
            # For many filings, scale based on complexity
            complexity_factor = min(1.0, total_complexity / (filing_count * 3))
            return max(2, min(self.max_workers, int(filing_count * complexity_factor)))

    def process_filings(
        self,
        filings: List[Dict[str, Any]],
        force_reprocess: bool = False,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process a list of filings in parallel.

        Args:
            filings: List of filings to process
            force_reprocess: Whether to force reprocessing of already processed filings

        Returns:
            Processing results
        """
        # Generate or use provided run_id
        if run_id is None:
            if self.run_id is None:
                self.run_id = generate_run_id()
            run_id = self.run_id
        else:
            self.run_id = run_id

        if not filings:
            logger.warning("No filings to process")
            log_etl_end(run_id, status="completed", error="No filings to process")
            return {
                "status": "no_filings",
                "stats": self.stats.to_dict(),
                "run_id": run_id,
            }

        # Update stats
        self.stats.total_filings = len(filings)

        # Log ETL start
        parameters = {
            "filings_count": len(filings),
            "force_reprocess": force_reprocess,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "rate_limit": self.rate_limiter_state["rate_limit"],
            "process_semantic": self.process_semantic,
            "process_quantitative": self.process_quantitative,
        }
        log_etl_start(run_id, parameters, f"Processing {len(filings)} filings")

        # Allocate workers based on filings
        workers = self._allocate_workers(filings)
        logger.info(f"Processing {len(filings)} filings with {workers} workers")

        # Process filings in parallel
        results = {"completed": [], "failed": [], "skipped": [], "run_id": run_id}

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit tasks
            future_to_filing = {}
            for filing in filings:
                # Check if filing has already been processed
                filing_id = filing.get("id", "unknown")
                if not force_reprocess and self.filing_processor.is_filing_processed(
                    filing_id
                ):
                    logger.info(f"Skipping already processed filing: {filing_id}")
                    results["skipped"].append(filing_id)
                    self.stats.skipped_filings += 1

                    # Log filing skipped
                    log_filing_processing(
                        run_id=run_id,
                        filing_id=filing_id,
                        company=filing.get("ticker", "unknown"),
                        filing_type=filing.get("type", "unknown"),
                        status="skipped",
                        processing_time=0.0,
                    )
                    continue

                # Submit task
                future = executor.submit(self._process_filing, filing, run_id)
                future_to_filing[future] = filing

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_filing):
                filing = future_to_filing[future]
                filing_id = filing.get("id", "unknown")

                try:
                    result = future.result()
                    if result.get("success", False):
                        results["completed"].append(filing_id)
                        self.stats.processed_filings += 1
                        logger.info(f"Successfully processed filing {filing_id}")

                        # Log filing processed
                        log_filing_processing(
                            run_id=run_id,
                            filing_id=filing_id,
                            company=filing.get("ticker", "unknown"),
                            filing_type=filing.get("type", "unknown"),
                            status="completed",
                            processing_time=result.get("processing_time", 0.0),
                        )
                    else:
                        results["failed"].append(filing_id)
                        self.stats.failed_filings += 1
                        error_msg = result.get("error", "Unknown error")
                        logger.error(
                            f"Failed to process filing {filing_id}: {error_msg}"
                        )

                        # Log filing failed
                        log_filing_processing(
                            run_id=run_id,
                            filing_id=filing_id,
                            company=filing.get("ticker", "unknown"),
                            filing_type=filing.get("type", "unknown"),
                            status="failed",
                            processing_time=result.get("processing_time", 0.0),
                            error=error_msg,
                        )
                except Exception as e:
                    results["failed"].append(filing_id)
                    self.stats.failed_filings += 1
                    error_msg = str(e)
                    logger.error(f"Error processing filing {filing_id}: {error_msg}")

                    # Log filing failed
                    log_filing_processing(
                        run_id=run_id,
                        filing_id=filing_id,
                        company=filing.get("ticker", "unknown"),
                        filing_type=filing.get("type", "unknown"),
                        status="failed",
                        processing_time=0.0,
                        error=error_msg,
                    )

        # Update embedding stats
        self.stats.embedding_stats = self.embedding_generator.token_usage
        self.stats.end_time = time.time()

        # Log embedding stats
        log_embedding_stats(
            run_id=run_id,
            tokens_used=self.embedding_generator.token_usage.get("total_tokens", 0),
            chunks_processed=len(filings),
            fallback_count=self.embedding_generator.token_usage.get("fallbacks", 0),
        )

        # Log ETL end
        log_etl_end(run_id, status="completed")

        # Add stats to results
        results["stats"] = self.stats.to_dict()

        return results

    def _process_filing(self, filing: Dict[str, Any], run_id: str) -> Dict[str, Any]:
        """Process a single filing with adaptive rate limiting.

        Args:
            filing: Filing metadata

        Returns:
            Processing result
        """
        filing_id = filing.get("id", "unknown")
        start_time = time.time()

        try:
            # Use phase timing context manager
            with log_phase_timing(run_id, f"filing_{filing_id}"):
                # Download filing content if needed
                if "content" not in filing:
                    with log_phase_timing(run_id, f"download_{filing_id}"):
                        with self.rate_limiter:
                            api_start = time.time()
                            try:
                                content = self.sec_downloader.download_filing_content(
                                    filing
                                )
                                api_time = time.time() - api_start
                                log_api_call(run_id, "SEC_API", True, api_time)

                                if not content:
                                    return {
                                        "success": False,
                                        "error": "Failed to download filing content",
                                        "processing_time": time.time() - start_time,
                                    }
                                filing["content"] = content
                            except Exception as e:
                                api_time = time.time() - api_start
                                log_api_call(run_id, "SEC_API", False, api_time, str(e))
                                raise

            # Create a filing-specific embedding generator with adaptive rate limiting
            with log_phase_timing(run_id, f"embedding_setup_{filing_id}"):
                filing_embedding_generator = RobustEmbeddingGenerator(
                    model=ConfigProvider.get_config(ETLConfig).embedding_model,
                    max_tokens_per_chunk=8000,
                    rate_limit=self.rate_limiter_state["rate_limit"],
                    filing_metadata=filing,
                    batch_size=self.batch_size,
                    max_retries=5,
                )

            # Process filing
            with log_phase_timing(run_id, f"process_{filing_id}"):
                result = self.filing_processor.process_filing(
                    filing,
                    embedding_generator=filing_embedding_generator,
                    process_semantic=self.process_semantic,
                    process_quantitative=self.process_quantitative,
                )

            # Update rate limiter based on embedding generation results
            if "embedding_metadata" in result:
                metadata = result["embedding_metadata"]
                if metadata.get("any_fallbacks", False):
                    # If we had fallbacks, slow down
                    old_rate = self.rate_limiter_state["rate_limit"]
                    self.rate_limiter.report_failure()
                    new_rate = self.rate_limiter_state["rate_limit"]

                    # Log rate limit adjustment
                    if old_rate != new_rate:
                        log_rate_limit_adjustment(
                            run_id=run_id,
                            old_rate=old_rate,
                            new_rate=new_rate,
                            reason=f"Fallbacks detected in filing {filing_id}",
                        )

            processing_time = time.time() - start_time
            return {
                "success": True,
                "result": result,
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Error processing filing {filing_id}: {str(e)}")
            processing_time = time.time() - start_time
            return {
                "success": False,
                "error": str(e),
                "processing_time": processing_time,
            }

    def process_company_filings(
        self,
        ticker: str,
        filing_types: List[str] = ["10-K", "10-Q", "8-K"],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None,
        force_reprocess: bool = False,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process filings for a specific company.

        Args:
            ticker: Company ticker symbol
            filing_types: Types of filings to process
            start_date: Start date for filings (YYYY-MM-DD)
            end_date: End date for filings (YYYY-MM-DD)
            limit: Maximum number of filings to process
            force_reprocess: Whether to force reprocessing of already processed filings

        Returns:
            Processing results
        """
        # Reset stats
        self.stats = ProcessingStats()

        # Generate or use provided run_id
        if run_id is None:
            self.run_id = generate_run_id()
            run_id = self.run_id
        else:
            self.run_id = run_id

        # Log ETL start for company
        parameters = {
            "ticker": ticker,
            "filing_types": filing_types,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit,
            "force_reprocess": force_reprocess,
        }
        log_etl_start(run_id, parameters, f"Processing filings for {ticker}")

        # Download filings
        with log_phase_timing(run_id, f"download_{ticker}"):
            api_start = time.time()
            try:
                with self.rate_limiter:
                    filings = self.sec_downloader.download_company_filings(
                        ticker=ticker,
                        filing_types=filing_types,
                        start_date=start_date,
                        end_date=end_date,
                        limit=limit,
                    )
                api_time = time.time() - api_start
                log_api_call(run_id, "SEC_API_COMPANY", True, api_time)
            except Exception as e:
                api_time = time.time() - api_start
                log_api_call(run_id, "SEC_API_COMPANY", False, api_time, str(e))
                log_company_processing(
                    run_id=run_id,
                    ticker=ticker,
                    status="failed",
                    error=f"Failed to download filings: {str(e)}",
                )
                log_etl_end(
                    run_id,
                    status="failed",
                    error=f"Failed to download filings for {ticker}: {str(e)}",
                )
                return {
                    "status": "error",
                    "error": f"Failed to download filings: {str(e)}",
                    "stats": self.stats.to_dict(),
                    "run_id": run_id,
                }

        if not filings:
            logger.warning(f"No filings found for {ticker}")
            log_company_processing(
                run_id=run_id, ticker=ticker, status="completed", filings_processed=0
            )
            log_etl_end(
                run_id, status="completed", error=f"No filings found for {ticker}"
            )
            return {
                "status": "no_filings",
                "stats": self.stats.to_dict(),
                "run_id": run_id,
            }

        # Process filings
        with log_phase_timing(run_id, f"process_{ticker}"):
            results = self.process_filings(filings, force_reprocess, run_id)
            results["ticker"] = ticker

        # Log company processing results
        log_company_processing(
            run_id=run_id,
            ticker=ticker,
            status="completed",
            filings_processed=len(results.get("completed", [])),
            filings_failed=len(results.get("failed", [])),
            filings_skipped=len(results.get("skipped", [])),
        )

        return results

    def process_multiple_companies(
        self,
        tickers: List[str],
        filing_types: List[str] = ["10-K", "10-Q", "8-K"],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit_per_company: Optional[int] = None,
        force_reprocess: bool = False,
        max_companies_in_parallel: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process filings for multiple companies.

        Args:
            tickers: List of company ticker symbols
            filing_types: Types of filings to process
            start_date: Start date for filings (YYYY-MM-DD)
            end_date: End date for filings (YYYY-MM-DD)
            limit_per_company: Maximum number of filings to process per company
            force_reprocess: Whether to force reprocessing of already processed filings
            max_companies_in_parallel: Maximum number of companies to process in parallel

        Returns:
            Processing results
        """
        # Generate or use provided run_id
        if run_id is None:
            self.run_id = generate_run_id()
            run_id = self.run_id
        else:
            self.run_id = run_id

        if not tickers:
            logger.warning("No tickers provided")
            log_etl_end(run_id, status="completed", error="No tickers provided")
            return {"status": "no_tickers", "run_id": run_id}

        # Determine how many companies to process in parallel
        if max_companies_in_parallel is None:
            # Default to a reasonable number based on max_workers
            max_companies_in_parallel = max(1, min(len(tickers), self.max_workers // 2))

        logger.info(
            f"Processing {len(tickers)} companies with up to {max_companies_in_parallel} in parallel"
        )

        # Log ETL start for multiple companies
        parameters = {
            "tickers": tickers,
            "filing_types": filing_types,
            "start_date": start_date,
            "end_date": end_date,
            "limit_per_company": limit_per_company,
            "force_reprocess": force_reprocess,
            "max_companies_in_parallel": max_companies_in_parallel,
        }
        log_etl_start(
            run_id, parameters, f"Processing filings for {len(tickers)} companies"
        )

        # Process companies in parallel
        results = {"companies": {}, "stats": {}, "run_id": run_id}
        overall_stats = ProcessingStats()

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_companies_in_parallel
        ) as executor:
            # Submit tasks
            future_to_ticker = {
                executor.submit(
                    self.process_company_filings,
                    ticker,
                    filing_types,
                    start_date,
                    end_date,
                    limit_per_company,
                    force_reprocess,
                    run_id,
                ): ticker
                for ticker in tickers
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]

                try:
                    result = future.result()
                    results["companies"][ticker] = result

                    # Update overall stats
                    company_stats = result.get("stats", {})
                    overall_stats.total_filings += company_stats.get("total_filings", 0)
                    overall_stats.processed_filings += company_stats.get(
                        "processed_filings", 0
                    )
                    overall_stats.failed_filings += company_stats.get(
                        "failed_filings", 0
                    )
                    overall_stats.skipped_filings += company_stats.get(
                        "skipped_filings", 0
                    )

                    logger.info(f"Completed processing for {ticker}")

                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Error processing company {ticker}: {error_msg}")
                    results["companies"][ticker] = {
                        "status": "error",
                        "error": error_msg,
                    }

                    # Log company processing error
                    log_company_processing(
                        run_id=run_id, ticker=ticker, status="failed", error=error_msg
                    )

        # Update overall stats
        overall_stats.end_time = time.time()
        results["stats"] = overall_stats.to_dict()

        # Add rate limiter stats
        results["rate_limiter_stats"] = self.rate_limiter.get_stats()

        # Log ETL end
        log_etl_end(run_id, status="completed")

        return results
