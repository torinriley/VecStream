#!/usr/bin/env python
"""
Run all VecStream tests.
"""

import os
import sys
import subprocess
import argparse

def run_tests(run_benchmarks=False):
    """Run all VecStream tests using pytest.
    
    Args:
        run_benchmarks: Whether to run benchmarks in addition to tests
    
    Returns:
        Exit code (0 if all tests pass, non-zero otherwise)
    """
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
    
    # Run benchmarks if requested
    if run_benchmarks:
        print("\n===== Running VecStream benchmarks =====\n")
        
        result_benchmarks = subprocess.run(
            ["python3", "tests/benchmarks/benchmark_vector_operations.py"],
            cwd=current_dir,
            capture_output=False
        )
        
        if result_benchmarks.returncode != 0:
            exit_code = result_benchmarks.returncode
    
    # Return exit code (0 if all tests pass, non-zero otherwise)
    return exit_code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VecStream tests")
    parser.add_argument(
        "--benchmarks", 
        action="store_true", 
        help="Run benchmarks in addition to tests"
    )
    args = parser.parse_args()
    
    sys.exit(run_tests(run_benchmarks=args.benchmarks)) 