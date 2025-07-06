import pytest
import subprocess
import sys
import os

# Get the absolute path to the test data file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
FBAS_JSON = os.path.join(TEST_DATA_DIR, 'top_tier.json')


def run_command(command):
    """Helper function to run a command and return its output."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True)
    return result.stdout


def test_check_intersection():
    """Test the check-intersection command."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "check-intersection"]
    output = run_command(command)
    assert "disjoint quorums" in output


def test_min_splitting_set():
    """Test the min-splitting-set command."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "min-splitting-set"]
    output = run_command(command)
    assert "Minimal splitting-set cardinality is" in output


def test_min_blocking_set():
    """Test the min-blocking-set command."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "min-blocking-set"]
    output = run_command(command)
    assert "Minimal blocking-set cardinality is" in output


def test_top_tier():
    """Test the top-tier command."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "top-tier"]
    output = run_command(command)
    assert "Top tier:" in output


def test_history_loss():
    """Test the history-loss command."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "history-loss"]
    output = run_command(command)
    assert "Minimal history-loss critical set cardinality is" in output


def test_min_quorum():
    """Test the min-quorum command."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "min-quorum"]
    output = run_command(command)
    assert "Example min quorum:" in output


if __name__ == '__main__':
    pytest.main([__file__])
