#!/usr/bin/env python
"""
Run all VecStream tests.
"""

import os
import sys
import subprocess

def run_tests():
    """Run all VecStream tests using pytest."""
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exit_code = 0
    
    # Print header
    print("\n===== Running VecStream unit tests =====\n")
    
    # Run unit tests
    result = subprocess.run(
        ["pytest", "-xvs", "tests/unit/"],
        cwd=current_dir,
        capture_output=False
    )
    
    if result.returncode != 0:
        exit_code = result.returncode
    
    # Run integration tests
    print("\n===== Running VecStream integration tests =====\n")
    
    result_integration = subprocess.run(
        ["pytest", "-xvs", "tests/integration/"],
        cwd=current_dir,
        capture_output=False
    )
    
    if result_integration.returncode != 0:
        exit_code = result_integration.returncode
    
    # Return exit code (0 if all tests pass, non-zero otherwise)
    return exit_code

if __name__ == "__main__":
    sys.exit(run_tests()) 