import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from dataclasses import fields
from python_fbas.config import Config, temporary_config, to_yaml


def _default_config_yaml():
    """A config file as a fresh `show-config` would emit it (i.e. defaults),
    independent of any process-wide config state left behind by other tests."""
    defaults = {f.name: getattr(Config(), f.name) for f in fields(Config)}
    with temporary_config(**defaults):
        return to_yaml()


README_PATH = Path(__file__).resolve().parents[1] / "README.md"
NETWORK_COMMANDS = {
    "check-intersection",
    "min-splitting-set",
    "min-blocking-set",
    "top-tier",
    "history-loss",
    "min-quorum",
    "max-scc",
    "to-json",
    "update-cache",
    "validator-metadata",
}

CONFIG_FILE_RE = re.compile(r"--config-file=(?P<q>['\"]?)(?P<path>[^\s'\"]+)(?P=q)")


def _parse_fenced_blocks(text):
    blocks = []
    in_block = False
    lang = None
    buf = []
    for line in text.splitlines():
        if line.startswith("```"):
            if not in_block:
                in_block = True
                lang = line.strip("`").strip()
                buf = []
            else:
                blocks.append((lang, "\n".join(buf)))
                in_block = False
                lang = None
                buf = []
            continue
        if in_block:
            buf.append(line)
    return blocks


def _is_relevant_block(block):
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return re.match(r"^\s*python-fbas\b", line) is not None
    return False


def _extract_fbas_value(line):
    match = re.search(r"--fbas=([^\s]+)", line)
    if not match:
        return None
    return match.group(1)


def _line_needs_network(line):
    if "python-fbas" not in line and "giulianolosa/python-fbas" not in line:
        return False
    fbas_value = _extract_fbas_value(line)
    if fbas_value:
        return fbas_value.startswith("http://") or fbas_value.startswith("https://")
    for cmd in NETWORK_COMMANDS:
        if re.search(rf"\b{re.escape(cmd)}\b", line):
            return True
    return False


def _block_needs_network(block):
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _line_needs_network(stripped):
            return True
    return False


def _line_needs_mount(line):
    # Relative --fbas paths are read from the working directory (mounted at
    # /work in docker). Relative --config-file inputs are handled separately:
    # the harness seeds them into the per-test tmpdir (see _relative_config_paths).
    fbas_value = _extract_fbas_value(line)
    if fbas_value and not (fbas_value.startswith("http://") or fbas_value.startswith("https://")):
        return True
    return False


def _block_needs_mount(block):
    return any(_line_needs_mount(line) for line in block.splitlines())


def _is_relative_path(path):
    return not path.startswith(("/", "$", "~"))


def _relative_config_paths(block):
    """Basenames of relative --config-file inputs referenced by the block.

    Such files are not created by the block itself (a README reader would have
    generated them in a previous step), so the harness seeds them into the
    per-test tmpdir before running.
    """
    names = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for match in CONFIG_FILE_RE.finditer(line):
            path = match.group("path")
            if _is_relative_path(path):
                names.append(os.path.basename(path))
    return names


def _rewrite_config_file(line, tmpdir):
    def repl(match):
        quote = match.group("q")
        path = match.group("path")
        if not _is_relative_path(path):
            return match.group(0)
        target = os.path.join(tmpdir, os.path.basename(path))
        return f"--config-file={quote}{target}{quote}"

    return CONFIG_FILE_RE.sub(repl, line)


def _ensure_mount_docker_line(line):
    if re.search(r"\s-v\s+.+:/work", line):
        return line
    if "docker run" not in line:
        return line
    return line.replace("docker run", 'docker run -v "$PWD:/work" -w /work', 1)


def _ensure_tmpdir_mount_docker_line(line, tmpdir):
    """Ensure a `docker run` line bind-mounts tmpdir at its own absolute path,
    so a seeded config file (referenced by absolute path) is readable inside
    the container."""
    if "docker run" not in line:
        return line
    spec = f'-v "{tmpdir}:{tmpdir}"'
    if spec in line:
        return line
    return line.replace("docker run", f"docker run {spec}", 1)


def _rewrite_docker_to_local(line):
    if "giulianolosa/python-fbas:latest" not in line:
        return line
    indent = re.match(r"^\s*", line).group(0)
    _, tail = line.split("giulianolosa/python-fbas:latest", 1)
    args = tail.strip()
    return f"{indent}python-fbas{(' ' + args) if args else ''}"


def _rewrite_redirections(line, tmpdir):
    def repl(match):
        op = match.group("op")
        path = match.group("path")
        if path.startswith(("/", "$")):
            return match.group(0)
        quote = ""
        if path[0] in ("'", '"') and path[-1] == path[0]:
            quote = path[0]
            path = path[1:-1]
        target = os.path.join(tmpdir, os.path.basename(path))
        return f"{op} {quote}{target}{quote}"

    return re.sub(r"(?P<op>>{1,2})\s*(?P<path>[^\s]+)", repl, line)


def _rewrite_block(block, mode, tmpdir, needs_mount):
    has_config = bool(_relative_config_paths(block))
    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        if mode == "docker":
            if "giulianolosa/python-fbas:latest" in line:
                new_line = _ensure_mount_docker_line(line) if needs_mount else line
                if has_config:
                    new_line = _ensure_tmpdir_mount_docker_line(new_line, tmpdir)
            elif re.match(r"^\s*python-fbas\b", line):
                prefix = "docker run --rm"
                if needs_mount:
                    prefix += ' -v "$PWD:/work" -w /work'
                if has_config:
                    # make the seeded config readable at its absolute path
                    prefix += f' -v "{tmpdir}:{tmpdir}"'
                prefix += " giulianolosa/python-fbas:latest"
                new_line = re.sub(r"^\s*python-fbas\b", prefix, line, 1)
            else:
                new_line = line
        else:
            new_line = _rewrite_docker_to_local(line)
        new_line = _rewrite_redirections(new_line, tmpdir)
        new_line = _rewrite_config_file(new_line, tmpdir)
        lines.append(new_line)
    return "\n".join(lines)


def _load_readme_blocks():
    text = README_PATH.read_text(encoding="utf-8")
    blocks = _parse_fenced_blocks(text)
    relevant = [(lang, block) for (lang, block) in blocks if _is_relevant_block(block)]
    return [block for (_, block) in relevant]


def _should_skip_block(block):
    if "<args>" in block:
        return "placeholder example"
    if os.environ.get("README_SKIP_NETWORK", "0") == "1" and _block_needs_network(block):
        return "README_SKIP_NETWORK enabled"
    return None


def _mode():
    mode = os.environ.get("README_MODE", "local").lower()
    if mode not in {"docker", "local"}:
        raise ValueError("README_MODE must be 'docker' or 'local'")
    return mode


def _run_block(block, mode, tmpdir):
    needs_mount = _block_needs_mount(block)
    # Seed any relative --config-file inputs the block expects to already exist
    # (the README shows generating them with `show-config` in a separate step).
    for name in _relative_config_paths(block):
        with open(os.path.join(tmpdir, name), "w", encoding="utf-8") as f:
            f.write(_default_config_yaml())
    rewritten = _rewrite_block(block, mode, tmpdir, needs_mount)
    print(f"\nREADME command (mode={mode}):\n{rewritten}\n", flush=True)
    script = "set -e\nset -o pipefail\n" + rewritten
    timeout = int(os.environ.get("README_CMD_TIMEOUT", "300"))
    result = subprocess.run(
        ["bash", "-lc", script],
        cwd=str(README_PATH.parent),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise AssertionError(
            "README command block failed\n"
            f"Mode: {mode}\n"
            f"Block:\n{block}\n\n"
            f"Rewritten:\n{rewritten}\n\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}\n"
        )


BLOCKS = _load_readme_blocks()
PARAMS = [
    pytest.param(
        block,
        id=f"block_{i + 1}",
        marks=[pytest.mark.network] if _block_needs_network(block) else [],
    )
    for i, block in enumerate(BLOCKS)
]


@pytest.mark.parametrize("block", PARAMS)
def test_readme_examples(block):
    mode = _mode()
    if mode == "docker" and not shutil.which("docker"):
        pytest.skip("docker not available")
    skip_reason = _should_skip_block(block)
    if skip_reason:
        pytest.skip(skip_reason)
    with tempfile.TemporaryDirectory(prefix="python-fbas-readme-") as tmpdir:
        _run_block(block, mode, tmpdir)


# Hermetic regression tests for the block-classification / seeding logic, so
# these gaps can't silently return without needing --run-network.

def test_commands_without_fbas_are_network_classified():
    """Every README command that loads FBAS data from the default source
    (i.e. has no --fbas) must be classified as needing network, so it is not
    silently exercised against the live default URL on a plain test run."""
    for block in BLOCKS:
        for line in block.splitlines():
            stripped = line.strip()
            if not re.match(r"^\s*python-fbas\b", stripped):
                continue
            if _extract_fbas_value(stripped):
                continue  # explicit --fbas: classification handled elsewhere
            if any(re.search(rf"\b{re.escape(cmd)}\b", stripped)
                   for cmd in NETWORK_COMMANDS):
                assert _block_needs_network(block), (
                    f"block loads default-source data but is not network-marked:\n{block}")


def test_relative_config_file_block_is_seeded_and_hermetic():
    """A README block that reads a relative --config-file must run without
    depending on any file outside its own tmpdir."""
    if not shutil.which("python-fbas"):
        pytest.skip("python-fbas console script not installed")
    # Same shape as the README "use config file" example, but reading a local
    # data file so we exercise config-file seeding without any network.
    block = ("python-fbas --config-file=my-config.yaml "
             "--fbas=tests/test_data/small/circular_1.json check-intersection")
    assert _relative_config_paths(block) == ["my-config.yaml"]
    with tempfile.TemporaryDirectory(prefix="python-fbas-readme-") as tmpdir:
        # _run_block asserts a zero exit code; a missing config would exit 1.
        _run_block(block, "local", tmpdir)


@pytest.mark.parametrize("block", [
    "python-fbas --config-file=my-config.yaml min-splitting-set",
    "docker run --rm giulianolosa/python-fbas:latest --config-file=my-config.yaml min-splitting-set",
])
def test_docker_mode_mounts_tmpdir_for_relative_config(block):
    """In docker mode, a rewritten (absolute-tmpdir) --config-file must be
    accompanied by a bind mount of that tmpdir, for both converted and
    pre-existing `docker run` lines."""
    tmpdir = "/tmp/python-fbas-readme-xyz"
    rewritten = _rewrite_block(block, "docker", tmpdir, _block_needs_mount(block))
    assert f"--config-file={tmpdir}/my-config.yaml" in rewritten
    assert f'-v "{tmpdir}:{tmpdir}"' in rewritten
