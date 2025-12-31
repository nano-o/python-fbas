import json
import pytest
import subprocess
import sys
import os

# Get the absolute path to the test data file
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
FBAS_JSON = os.path.join(TEST_DATA_DIR, 'small', 'top_tier.json')


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


def test_min_splitting_set_groups():
    """Test the min-splitting-set command with group-by flag."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "--group-by=homeDomain",
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


def test_min_blocking_set_groups():
    """Test the min-blocking-set command with group-by flag."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "--group-by=homeDomain",
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


def test_top_tier_from_validator():
    """Test the top-tier command with --from-validator option."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "top-tier",
        "--from-validator=GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH"]
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


def test_max_scc():
    """Test the max-scc command."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "max-scc"]
    output = run_command(command)
    assert "Maximal SCC with a quorum:" in output


def test_validator_display_id():
    """Test the --validator-display=id command-line option."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "--validator-display=id",
        "top-tier"
    ]
    output = run_command(command)
    assert "GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH" in output
    assert "SDF 1" not in output


def test_validator_display_name():
    """Test the --validator-display=name command-line option."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "--validator-display=name",
        "top-tier"
    ]
    output = run_command(command)
    assert "GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH" not in output
    assert "SDF 1" in output


def test_validator_display_both():
    """Test the --validator-display=both command-line option."""
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        f"--fbas={FBAS_JSON}",
        "--validator-display=both",
        "top-tier"
    ]
    output = run_command(command)
    assert "GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH" in output
    assert "SDF 1" in output


def test_random_sybil_attack_two_clusters_smoke(tmp_path):
    """Test random-sybil-attack-fbas with two Sybil clusters."""
    generator_config = tmp_path / "generator.yaml"
    generator_config.write_text(
        "\n".join([
            "orgs: 8",
            "sybils: 4",
            "sybils_cluster_2: 4",
            "num_sybil_clusters: 2",
            "sybil_bridge_orgs: 2",
            "original_edge_probability: 0.6",
            "sybil_sybil_edge_probability: 0.6",
            "sybil2_sybil2_edge_probability: 0.6",
            "attacker_to_sybil_edge_probability: 0.8",
            "attacker_to_attacker_edge_probability: 0.2",
            "attacker_to_honest_edge_probability: 0.2",
            "sybil_to_honest_edge_probability: 0.1",
            "sybil_to_attacker_edge_probability: 0.1",
            "sybil_to_sybil_bridge_edge_probability: 0.8",
            "sybil_bridge_to_sybil2_edge_probability: 0.8",
            "sybil_bridge_to_sybil_bridge_edge_probability: 0.1",
            "sybil2_to_honest_edge_probability: 0.1",
            "sybil2_to_attacker_edge_probability: 0.1",
            "sybil2_to_sybil1_edge_probability: 0.1",
            "sybil2_to_sybil_bridge_edge_probability: 0.1",
            "sybil1_to_sybil2_edge_probability: 0.1",
            "connect_attacker_to_attacker: false",
            "connect_attacker_to_honest: false",
            "connect_sybil_to_honest: false",
            "connect_sybil_to_attacker: false",
            "connect_sybil_bridge_to_sybil_bridge: false",
            "connect_sybil2_to_honest: false",
            "connect_sybil2_to_attacker: false",
            "connect_sybil2_to_sybil1: false",
            "connect_sybil2_to_sybil_bridge: false",
            "connect_sybil1_to_sybil2: false",
            "seed: 1",
        ]),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "python_fbas.main",
        "random-sybil-attack-fbas",
        "--print-fbas",
        f"--generator-config={generator_config}",
        f"--runs-dir={tmp_path / 'runs'}",
    ]
    output = run_command(command)
    data = json.loads(output)
    assert isinstance(data, list)
    assert data
    assert isinstance(data[0], dict)
    run_root = tmp_path / "runs"
    run_dirs = [path for path in run_root.iterdir() if path.is_dir()]
    assert run_dirs
    run_dir = run_dirs[0]
    assert (run_dir / "python-fbas.generator.cfg").exists()
    assert (run_dir / "python-fbas.sybil-detection.cfg").exists()


if __name__ == '__main__':
    pytest.main([__file__])
