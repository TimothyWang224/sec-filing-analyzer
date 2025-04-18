# Optimized Parallel Processing Framework

This document describes the optimized parallel processing framework for the SEC Filing Analyzer, which efficiently processes multiple filings while respecting API limits.

## Overview

The optimized parallel processing framework is designed to maximize throughput while avoiding rate limit errors from the OpenAI API. It includes:

1. **Adaptive Rate Limiter**: Dynamically adjusts rate limits based on API response patterns
2. **Optimized Parallel Processor**: Efficiently processes multiple filings in parallel
3. **Dynamic Worker Allocation**: Allocates workers based on filing complexity
4. **Comprehensive Error Handling**: Robust error recovery mechanisms

## Components

### Adaptive Rate Limiter

The `AdaptiveRateLimiter` class provides dynamic rate limiting based on API response patterns:

- **Dynamic Adjustment**: Automatically adjusts rate limits based on success/failure patterns
- **Shared State**: Coordinates rate limiting across multiple threads
- **Exponential Backoff**: Slows down exponentially on failures
- **Gradual Speedup**: Gradually increases throughput on sustained success

```python
from sec_filing_analyzer.utils.adaptive_rate_limiter import AdaptiveRateLimiter

# Create a rate limiter
rate_limiter = AdaptiveRateLimiter(
    initial_rate_limit=0.5,  # Start conservative
    min_rate_limit=0.1,      # Fastest we'll ever go
    max_rate_limit=5.0       # Slowest we'll ever go
)

# Use as a context manager
with rate_limiter:
    # Make API call
    response = api.call()

# Or use manually
rate_limiter.wait()
try:
    # Make API call
    response = api.call()
    rate_limiter.report_success()
except Exception as e:
    rate_limiter.report_failure()
```

### Optimized Parallel Processor

The `OptimizedParallelProcessor` class efficiently processes multiple filings in parallel:

- **Adaptive Rate Limiting**: Uses the adaptive rate limiter to avoid rate limit errors
- **Dynamic Worker Allocation**: Allocates workers based on filing complexity
- **Efficient Batching**: Batches embedding requests for optimal throughput
- **Comprehensive Error Handling**: Robust error recovery mechanisms

```python
from sec_filing_analyzer.pipeline.optimized_parallel_processor import OptimizedParallelProcessor

# Create a processor
processor = OptimizedParallelProcessor(
    max_workers=4,
    initial_rate_limit=0.5,
    batch_size=50,
    process_semantic=True,
    process_quantitative=True
)

# Process filings for multiple companies
results = processor.process_multiple_companies(
    tickers=["AAPL", "MSFT", "NVDA"],
    filing_types=["10-K", "10-Q", "8-K"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    limit_per_company=5,
    force_reprocess=False,
    max_companies_in_parallel=2
)
```

## Parallelization Strategy

The framework uses a multi-level parallelization strategy:

1. **Company-Level Parallelism**: Process multiple companies in parallel
2. **Filing-Level Parallelism**: Process multiple filings for each company in parallel
3. **Batch-Level Parallelism**: Process batches of text chunks efficiently

Each level of parallelism is controlled by parameters:

- `max_companies_in_parallel`: Controls company-level parallelism
- `max_workers`: Controls filing-level parallelism
- `batch_size`: Controls batch-level parallelism

## Rate Limiting Strategy

The framework uses a sophisticated rate limiting strategy:

1. **Adaptive Rate Limiting**: Dynamically adjusts rate limits based on API response patterns
2. **Shared State**: Coordinates rate limiting across multiple threads
3. **Exponential Backoff**: Slows down exponentially on failures
4. **Gradual Speedup**: Gradually increases throughput on sustained success

## Usage

### Basic Usage

```python
from sec_filing_analyzer.pipeline.optimized_parallel_processor import OptimizedParallelProcessor

# Create a processor
processor = OptimizedParallelProcessor()

# Process filings for a single company
results = processor.process_company_filings(
    ticker="AAPL",
    filing_types=["10-K", "10-Q", "8-K"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    limit=5
)
```

### Advanced Usage

```python
from sec_filing_analyzer.pipeline.optimized_parallel_processor import OptimizedParallelProcessor

# Create a processor with custom parameters
processor = OptimizedParallelProcessor(
    max_workers=8,
    initial_rate_limit=0.2,
    batch_size=100,
    process_semantic=True,
    process_quantitative=False
)

# Process filings for multiple companies
results = processor.process_multiple_companies(
    tickers=["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"],
    filing_types=["10-K", "10-Q"],
    start_date="2023-01-01",
    end_date="2023-12-31",
    limit_per_company=2,
    force_reprocess=True,
    max_companies_in_parallel=3
)
```

### Command-Line Usage

The framework includes a command-line script for testing:

```bash
python test_optimized_parallel_processor.py \
    --tickers AAPL MSFT NVDA \
    --filing-types 10-K 10-Q \
    --days 365 \
    --limit 5 \
    --max-workers 8 \
    --batch-size 100 \
    --rate-limit 0.2 \
    --max-companies 3 \
    --semantic-only \
    --output results.json
```

## Performance Tuning

The framework can be tuned for optimal performance:

### For Maximum Throughput

```python
processor = OptimizedParallelProcessor(
    max_workers=8,
    initial_rate_limit=0.2,
    batch_size=100
)
```

### For Maximum Reliability

```python
processor = OptimizedParallelProcessor(
    max_workers=4,
    initial_rate_limit=1.0,
    batch_size=50
)
```

### For Balanced Performance

```python
processor = OptimizedParallelProcessor(
    max_workers=6,
    initial_rate_limit=0.5,
    batch_size=75
)
```

## Monitoring and Statistics

The framework provides comprehensive monitoring and statistics:

```python
# Get processing statistics
stats = results["stats"]
print(f"Processed {stats['processed_filings']} filings in {stats['duration_formatted']}")
print(f"Success rate: {stats['success_rate']}")
print(f"Processing rate: {stats['processing_rate']}")

# Get rate limiter statistics
rate_limiter_stats = results["rate_limiter_stats"]
print(f"Current rate limit: {rate_limiter_stats['current_rate_limit']:.2f}s")
print(f"Total requests: {rate_limiter_stats['total_requests']}")
print(f"Total failures: {rate_limiter_stats['total_failures']}")
```

## Error Handling

The framework includes comprehensive error handling:

- **Automatic Retries**: Automatically retries failed API calls
- **Fallback Mechanisms**: Uses fallback mechanisms for failed embeddings
- **Detailed Logging**: Provides detailed logging for debugging
- **Error Recovery**: Recovers from errors and continues processing

## Conclusion

The optimized parallel processing framework provides an efficient and reliable way to process multiple SEC filings while respecting API limits. It balances throughput with reliability, making it ideal for large-scale processing tasks.
