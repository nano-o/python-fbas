import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


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
}


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
    fbas_value = _extract_fbas_value(line)
    if fbas_value and not (fbas_value.startswith("http://") or fbas_value.startswith("https://")):
        return True
    config_match = re.search(r"--config-file=([^\s]+)", line)
    if config_match:
        path = config_match.group(1)
        if not path.startswith("/"):
            return True
    return False


def _block_needs_mount(block):
    return any(_line_needs_mount(line) for line in block.splitlines())


def _ensure_mount_docker_line(line):
    if re.search(r"\s-v\s+.+:/work", line):
        return line
    if "docker run" not in line:
        return line
    return line.replace("docker run", 'docker run -v "$PWD:/work" -w /work', 1)


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
    lines = []
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        if mode == "docker":
            if "giulianolosa/python-fbas:latest" in line:
                new_line = _ensure_mount_docker_line(line) if needs_mount else line
            elif re.match(r"^\s*python-fbas\b", line):
                prefix = "docker run --rm"
                if needs_mount:
                    prefix += ' -v "$PWD:/work" -w /work'
                prefix += " giulianolosa/python-fbas:latest"
                new_line = re.sub(r"^\s*python-fbas\b", prefix, line, 1)
            else:
                new_line = line
        else:
            new_line = _rewrite_docker_to_local(line)
        new_line = _rewrite_redirections(new_line, tmpdir)
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


@pytest.mark.parametrize("block", BLOCKS, ids=[f"block_{i+1}" for i in range(len(BLOCKS))])
def test_readme_examples(block):
    mode = _mode()
    if mode == "docker" and not shutil.which("docker"):
        pytest.skip("docker not available")
    skip_reason = _should_skip_block(block)
    if skip_reason:
        pytest.skip(skip_reason)
    with tempfile.TemporaryDirectory(prefix="python-fbas-readme-") as tmpdir:
        _run_block(block, mode, tmpdir)
