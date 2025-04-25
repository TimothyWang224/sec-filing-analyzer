"""
Tool Memoization

This module provides a decorator for caching tool results to prevent redundant tool calls.
"""

import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global registry of memoized functions
_memoized_functions = []


def _make_hashable(obj):
    """
    Convert an object to a hashable representation.

    Args:
        obj: The object to convert

    Returns:
        A hashable representation of the object
    """
    if isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    elif isinstance(obj, (list, tuple)):
        return tuple(_make_hashable(x) for x in obj)
    elif isinstance(obj, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in obj.items()))
    else:
        return str(obj)


def memoize_tool(func):
    """
    Decorator to memoize tool results.

    Args:
        func: The async function to memoize

    Returns:
        Memoized async function
    """
    # Dictionary to store cached results
    cache = {}

    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Get the tool name for logging
        tool_name = args[0].name if args and hasattr(args[0], "name") else func.__name__

        # Create a hashable key from the arguments
        try:
            args_key = _make_hashable(args)
            kwargs_key = _make_hashable(kwargs)
            cache_key = (args_key, kwargs_key)

            # Check if result is in cache
            if cache_key in cache:
                logger.info(f"Cache hit for tool '{tool_name}'")
                return cache[cache_key]

            # Cache miss - execute the function
            logger.info(f"Cache miss for tool '{tool_name}'")
            result = await func(*args, **kwargs)

            # Store the result in the cache
            cache[cache_key] = result

            return result
        except Exception as e:
            # If there's an error in caching (e.g., unhashable type),
            # log it and fall back to executing the function without caching
            logger.warning(f"Error in tool caching: {str(e)}. Executing without cache.")
            return await func(*args, **kwargs)

    # Register the function for cache clearing
    _memoized_functions.append((func.__name__, wrapper, cache))

    return wrapper


def clear_tool_caches():
    """Clear all tool caches."""
    for func_name, _, cache in _memoized_functions:
        cache.clear()
        logger.info(f"Cleared cache for {func_name}")

    logger.info(f"Cleared caches for {len(_memoized_functions)} tools")
