import asyncio
import json
import logging
from src.tools.sec_financial_data import SECFinancialDataTool
from src.tools.tool_parameter_helper import _validate_sec_financial_data_parameters

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def debug_sec_financial_data_tool():
    """Debug the SEC Financial Data Tool."""
    print("\n\n" + "="*80)
    print("DEBUGGING SEC FINANCIAL DATA TOOL")
    print("="*80 + "\n")
    
    # Create the tool
    tool = SECFinancialDataTool()
    
    # Test case 1: Parameters in the correct format
    print("\nTest Case 1: Parameters in the correct format")
    params1 = {
        "query_type": "metrics",
        "parameters": {
            "ticker": "AAPL"
        }
    }
    
    # Validate parameters
    validation_result1 = _validate_sec_financial_data_parameters(params1)
    print(f"Validation Result: {json.dumps(validation_result1, indent=2)}")
    
    # Test validate_args method
    is_valid1 = tool.validate_args(**params1)
    print(f"Is Valid: {is_valid1}")
    
    # Test case 2: Parameters in the wrong format (missing nested parameters)
    print("\nTest Case 2: Parameters in the wrong format (missing nested parameters)")
    params2 = {
        "query_type": "metrics",
        "ticker": "AAPL"  # This should be inside a "parameters" dictionary
    }
    
    # Validate parameters
    validation_result2 = _validate_sec_financial_data_parameters(params2)
    print(f"Validation Result: {json.dumps(validation_result2, indent=2)}")
    
    # Test validate_args method
    is_valid2 = tool.validate_args(**params2)
    print(f"Is Valid: {is_valid2}")
    
    # Test case 3: Parameters in the format used in the test_tool_contract.py script
    print("\nTest Case 3: Parameters in the format used in the test_tool_contract.py script")
    params3 = {
        "query_type": "metrics",
        "ticker": "AAPL",
        "year": 2022
    }
    
    # Validate parameters
    validation_result3 = _validate_sec_financial_data_parameters(params3)
    print(f"Validation Result: {json.dumps(validation_result3, indent=2)}")
    
    # Test validate_args method
    is_valid3 = tool.validate_args(**params3)
    print(f"Is Valid: {is_valid3}")
    
    # Test case 4: Execute the tool with fixed parameters
    print("\nTest Case 4: Execute the tool with fixed parameters")
    fixed_params = validation_result3["parameters"]
    
    try:
        result = await tool.execute(**fixed_params)
        print(f"Execution Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Execution Error: {str(e)}")
    
    # Test case 5: Execute the tool with parameters in the correct format
    print("\nTest Case 5: Execute the tool with parameters in the correct format")
    try:
        result = await tool.execute(**params1)
        print(f"Execution Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        print(f"Execution Error: {str(e)}")
    
    return "Debug complete"

if __name__ == "__main__":
    asyncio.run(debug_sec_financial_data_tool())
