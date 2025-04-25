"""
Check Edgar Package Script

This script checks what's available in the edgar package.
"""

import inspect
import sys

# Try to import edgar
try:
    import edgar

    print(f"Edgar package version: {edgar.__version__ if hasattr(edgar, '__version__') else 'Unknown'}")
    print(f"Edgar package path: {edgar.__file__}")

    # Print all modules and attributes
    print("\nModules and attributes in edgar package:")
    for name in dir(edgar):
        if not name.startswith("__"):
            obj = getattr(edgar, name)
            if inspect.ismodule(obj):
                print(f"  Module: {name}")
                for subname in dir(obj):
                    if not subname.startswith("__"):
                        print(f"    {subname}")
            else:
                print(f"  {name}: {type(obj).__name__}")

    # Try to import Company
    try:
        from edgar import Company

        print("\nCompany class is available")
        print(f"Company class path: {inspect.getfile(Company)}")
    except ImportError as e:
        print(f"\nCannot import Company: {e}")

    # Try to import XBRL
    try:
        from edgar import XBRL

        print("\nXBRL class is available")
        print(f"XBRL class path: {inspect.getfile(XBRL)}")
    except ImportError as e:
        print(f"\nCannot import XBRL: {e}")

    # Check edgar.xbrl module
    try:
        import edgar.xbrl

        print("\nedgar.xbrl module is available")
        print(f"edgar.xbrl module path: {edgar.xbrl.__file__}")

        # Print all attributes in edgar.xbrl
        print("\nAttributes in edgar.xbrl module:")
        for name in dir(edgar.xbrl):
            if not name.startswith("__"):
                obj = getattr(edgar.xbrl, name)
                if inspect.isclass(obj):
                    print(f"  Class: {name}")
                elif inspect.isfunction(obj):
                    print(f"  Function: {name}")
                else:
                    print(f"  {name}: {type(obj).__name__}")
    except ImportError as e:
        print(f"\nCannot import edgar.xbrl: {e}")

except ImportError as e:
    print(f"Cannot import edgar: {e}")
    sys.exit(1)
