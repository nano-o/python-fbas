#!/usr/bin/env python3
"""
Test runner script that suppresses output when all tests pass,
but shows full output on first failure.
"""

import subprocess
import sys
import re


def run_tests():
    """Run pytest and process output."""
    # Run pytest with -x (stop on first failure) and capture output
    cmd = ["python3", "-m", "pytest", "tests/", "-x", "--tb=short"]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300)
        output = result.stdout
        stderr = result.stderr

        # Check if all tests passed by looking for the summary line
        if result.returncode == 0:
            # All tests passed - extract just the summary
            lines = output.split('\n')
            summary_line = None

            # Look for the summary line (e.g., "156 passed in 17.81s")
            for line in lines:
                if re.search(r'\d+ passed',
                             line) and 'in' in line and 's' in line:
                    summary_line = line.strip()
                    break

            if summary_line:
                print(f"✅ ALL TESTS PASSED: {summary_line}")
            else:
                print("✅ ALL TESTS PASSED")

            return 0
        else:
            # Some tests failed - show full output
            print("❌ TESTS FAILED:")
            print("=" * 60)
            if stderr:
                print("STDERR:")
                print(stderr)
                print("=" * 60)
            print("STDOUT:")
            print(output)
            return result.returncode

    except subprocess.TimeoutExpired:
        print("❌ TESTS TIMED OUT (300 seconds)")
        return 1
    except Exception as e:
        print(f"❌ ERROR RUNNING TESTS: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
