#!/usr/bin/env python3
"""
Benchmark script comparing python-fbas vs Rust fbas_analyzer on all available test data.

This script runs comprehensive benchmarks on all FBAS test files in tests/test_data/random/
and generates markdown and CSV reports comparing the performance of both tools.
"""

import json
import subprocess
import time
import signal
import csv
import argparse
from pathlib import Path
from datetime import datetime
import sys
import os
import tempfile
import shlex
import shutil
from python_fbas.serialization import deserialize, serialize, detect_format


class TimeoutError(Exception):
    pass




def timeout_handler(signum, frame):
    raise TimeoutError("Command timed out")


def run_with_timeout(cmd, timeout=10):
    """Run command with timeout and return result."""
    start_time = time.time()
    process = None

    try:
        # Start the process
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            preexec_fn=os.setsid)  # Create new process group

        # Wait for completion or timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            elapsed = time.time() - start_time

            return {
                'success': process.returncode == 0,
                'elapsed': elapsed,
                'stdout': stdout,
                'stderr': stderr,
                'timed_out': False
            }
        except subprocess.TimeoutExpired:
            # Kill the entire process group to ensure cleanup
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(0.1)  # Give it a moment to terminate gracefully
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass  # Process already dead

            # Clean up
            try:
                process.kill()
                process.wait()
            except (OSError, ProcessLookupError):
                pass

            return {
                'success': False,
                'elapsed': timeout,
                'stdout': '',
                'stderr': 'TIMEOUT',
                'timed_out': True
            }
    except Exception as e:
        # Ensure process is killed if something goes wrong
        if process:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.kill()
            except (OSError, ProcessLookupError):
                pass

        return {
            'success': False,
            'elapsed': time.time() - start_time,
            'stdout': '',
            'stderr': str(e),
            'timed_out': False
        }


def get_network_info(file_path):
    """Extract organization count from filename."""
    name = file_path.name

    # Extract organization count from filename
    if "_orgs" in name:
        parts = name.split("_")
        for i, part in enumerate(parts):
            if part == "orgs" and i > 0:
                try:
                    org_count = int(parts[i - 1])
                    return org_count
                except ValueError:
                    pass

    # Fallback: try to extract from other patterns
    if "almost_symmetric_network_" in name:
        parts = name.split("_")
        for i, part in enumerate(parts):
            if part.isdigit():
                try:
                    return int(part)
                except ValueError:
                    pass

    # Last resort: return 0 if can't parse
    return 0


def run_benchmark_on_file(
        original_file,
        timeout,
        python_fbas_cmd,
        rust_tool_path):
    """Run all benchmark tests on a single file pair."""
    print(f"  Testing {original_file.name}...", end="", flush=True)

    org_count = get_network_info(original_file)
    
    # Create temporary file for stellarbeat format
    temp_file = None
    try:
        # Load the original file
        with open(original_file, 'r') as f:
            original_data = json.load(f)
        
        format_type = detect_format(original_data)
        if format_type == 'stellarbeat':
            # Already in array format, use as-is
            stellarbeat_json = json.dumps(original_data, indent=2)
        elif format_type == 'python-fbas':
            # Convert python-fbas format to stellarbeat for fbas_analyzer
            graph = deserialize(json.dumps(original_data))
            stellarbeat_json = serialize(graph, format='stellarbeat')
        else:
            raise ValueError(
                f"Unknown FBAS JSON format in {original_file.name}")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tf:
            tf.write(stellarbeat_json)
            temp_file = tf.name

        # Define test cases - use original file for python, temp file for rust
        test_cases = [{'name': 'intersection',
                       'python_cmd': f'{python_fbas_cmd} --fbas={original_file} check-intersection',
                       'rust_cmd': f'{rust_tool_path} {temp_file} --alternative-quorum-intersection-check --results-only'},
                      {'name': 'blocking',
                       'python_cmd': f'{python_fbas_cmd} --fbas={original_file} min-blocking-set',
                       'rust_cmd': f'{rust_tool_path} {temp_file} -b --results-only'},
                      {'name': 'splitting',
                       'python_cmd': f'{python_fbas_cmd} --fbas={original_file} min-splitting-set',
                       'rust_cmd': f'{rust_tool_path} {temp_file} -s --results-only'}]

        results = []

        for test_case in test_cases:
            # Run Python version
            py_result = run_with_timeout(test_case['python_cmd'], timeout)
            py_status = 'timeout' if py_result['timed_out'] else (
                'success' if py_result['success'] else 'error')
            py_time = py_result['elapsed'] if py_status == 'success' else None

            # Run Rust version
            rust_result = run_with_timeout(test_case['rust_cmd'], timeout)
            rust_status = 'timeout' if rust_result['timed_out'] else (
                'success' if rust_result['success'] else 'error')
            rust_time = rust_result['elapsed'] if rust_status == 'success' else None

            # Calculate speedup
            speedup = None
            if py_time and rust_time:
                speedup = py_time / rust_time

            results.append({
                'file_name': original_file.name,
                'network_size': org_count,
                'test_type': test_case['name'],
                'python_time': py_time,
                'python_status': py_status,
                'rust_time': rust_time,
                'rust_status': rust_status,
                'speedup': speedup
            })

        # Print simple progress indicator
        # Count individual tool runs (each result has both python and rust runs)
        # Each result represents 1 python + 1 rust run
        total_runs = len(results) * 2
        completed_runs = sum(
            1 for r in results if r['python_status'] != 'error') + sum(
            1 for r in results if r['rust_status'] != 'error')
        print(f" {completed_runs}/{total_runs} completed")

        return results
        
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except Exception:
                pass  # Ignore cleanup errors


def generate_short_handle(file_name, handle_map):
    """Generate a short handle for a file name."""
    # Extract key parts from filename
    name = file_name.replace('.json', '')

    # Handle different patterns
    if 'almost_symmetric_network_' in name:
        parts = name.split('_')
        # Find the orgs number
        for i, part in enumerate(parts):
            if part == 'orgs' and i > 0:
                orgs_num = parts[i - 1]
                # Check if it has delete_prob_factor
                if 'delete_prob_factor_' in name:
                    # Find the factor number
                    for j, p in enumerate(parts):
                        if p == 'factor' and j > 0 and parts[j - 1] == 'prob':
                            factor_num = parts[j + 1]
                            handle = f"N{orgs_num}F{factor_num}"
                            break
                    else:
                        handle = f"N{orgs_num}"
                else:
                    handle = f"N{orgs_num}"
                break
        else:
            # Fallback
            handle = f"F{len(handle_map) + 1}"
    else:
        # Simple fallback for other patterns
        handle = f"F{len(handle_map) + 1}"

    # Ensure uniqueness
    base_handle = handle
    counter = 1
    while handle in handle_map.values():
        handle = f"{base_handle}_{counter}"
        counter += 1

    return handle


def generate_markdown_report(results, timeout, output_file):
    """Generate markdown report with results."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Calculate summary statistics
    total_tests = len(results)
    python_successes = sum(
        1 for r in results if r['python_status'] == 'success')
    rust_successes = sum(1 for r in results if r['rust_status'] == 'success')
    both_success = sum(
        1 for r in results if r['python_status'] == 'success' and r['rust_status'] == 'success')

    # Group results by file for table and create handle mapping
    files_data = {}
    handle_map = {}
    for result in results:
        file_name = result['file_name']
        if file_name not in files_data:
            files_data[file_name] = {
                'network_size': result['network_size'],
                'tests': {}
            }
            handle_map[file_name] = generate_short_handle(
                file_name, handle_map)
        files_data[file_name]['tests'][result['test_type']] = result

    with open(output_file, 'w') as f:
        f.write("# FBAS Tools Benchmark Results\n\n")
        f.write(f"**Generated:** {timestamp}\n")
        f.write(f"**Timeout:** {timeout} seconds per test\n\n")

        f.write("## Commands Used\n\n")
        f.write("### Python-fbas\n")
        f.write("```bash\n")
        f.write("python-fbas --fbas=<file> check-intersection\n")
        f.write("python-fbas --fbas=<file> min-blocking-set\n")
        f.write("python-fbas --fbas=<file> min-splitting-set\n")
        f.write("```\n\n")

        f.write("### Rust fbas_analyzer\n")
        f.write("```bash\n")
        f.write(
            "./fbas_analyzer <file> --alternative-quorum-intersection-check --results-only\n")
        f.write("./fbas_analyzer <file> -b --results-only\n")
        f.write("./fbas_analyzer <file> -s --results-only\n")
        f.write("```\n\n")
        f.write(
            "**Note:** Test files are automatically converted from python-fbas format to stellarbeat format for fbas_analyzer.\n\n")

        f.write("## Summary\n\n")
        f.write(f"- **Total tests:** {total_tests}\n")
        f.write(f"- **Python successful:** {python_successes}/{total_tests}\n")
        f.write(f"- **Rust successful:** {rust_successes}/{total_tests}\n")
        f.write(f"- **Both successful:** {both_success}/{total_tests}\n\n")

        f.write("## Results\n\n")

        # Sort files by network size, then by name
        sorted_files = sorted(
            files_data.items(), key=lambda x: (
                x[1]['network_size'], x[0]))

        # Calculate column widths for alignment
        rows = []
        for file_name, file_data in sorted_files:
            # This is actually org_count from the data structure
            org_count = file_data['network_size']
            tests = file_data['tests']
            handle = handle_map[file_name]

            def format_time(test_type):
                if test_type not in tests:
                    return "N/A", "N/A"

                test = tests[test_type]
                py_str = f"{test['python_time']:.3f}s" if test['python_time'] else test['python_status']
                rust_str = f"{test['rust_time']:.3f}s" if test['rust_time'] else test['rust_status']
                return py_str, rust_str

            int_py, int_rust = format_time('intersection')
            block_py, block_rust = format_time('blocking')
            split_py, split_rust = format_time('splitting')

            rows.append([handle, str(org_count), int_py, int_rust,
                        block_py, block_rust, split_py, split_rust])

        # Calculate column widths
        headers = [
            "Handle",
            "Orgs",
            "Python",
            "Rust",
            "Python",
            "Rust",
            "Python",
            "Rust"]
        subheaders = [
            "",
            "",
            "Intersection",
            "",
            "Blocking",
            "",
            "Splitting",
            ""]

        # Find maximum width for each column
        col_widths = []
        for i in range(8):
            max_width = max(
                len(headers[i]),
                len(subheaders[i]),
                max(len(row[i]) for row in rows) if rows else 0
            )
            col_widths.append(max_width)

        # Write aligned table
        def write_row(cells):
            padded_cells = [cell.ljust(col_widths[i])
                            for i, cell in enumerate(cells)]
            f.write(f"| {' | '.join(padded_cells)} |\n")

        # Header row
        write_row(["Handle", "Orgs", "Intersection",
                  "", "Blocking", "", "Splitting", ""])

        # Separator row
        separators = ["-" * width for width in col_widths]
        f.write(f"|{':'.join(f'-{sep}-' for sep in separators)}|\n")

        # Subheader row
        write_row(["", "", "Python", "Rust",
                  "Python", "Rust", "Python", "Rust"])

        # Data rows
        for row in rows:
            write_row(row)

        # Add mapping section
        f.write("\n## File Handle Mapping\n\n")

        # Sort handles for better readability
        sorted_handles = sorted(handle_map.items(), key=lambda x: x[1])

        # Calculate column widths for mapping table
        if sorted_handles:
            handle_width = max(len("Handle"), max(len(handle)
                               for _, handle in sorted_handles))
            filename_width = max(
                len("File Name"), max(
                    len(filename) for filename, _ in sorted_handles))

            # Write mapping table with proper alignment
            f.write(
                f"| {'Handle'.ljust(handle_width)} | {'File Name'.ljust(filename_width)} |\n")
            f.write(
                f"|{'-' * (handle_width + 2)}|{'-' * (filename_width + 2)}|\n")

            for file_name, handle in sorted_handles:
                f.write(
                    f"| {handle.ljust(handle_width)} | {file_name.ljust(filename_width)} |\n")


def generate_csv_report(results, output_file):
    """Generate CSV report with detailed results."""
    fieldnames = [
        'file_name',
        'network_size',
        'test_type',
        'python_time',
        'python_status',
        'rust_time',
        'rust_status',
        'speedup']

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def cleanup_remaining_processes():
    """Kill any remaining fbas_analyzer or python-fbas processes."""
    try:
        # Kill any remaining fbas_analyzer processes
        subprocess.run(['pkill', '-f', 'fbas_analyzer'], capture_output=True)
        subprocess.run(['pkill', '-f', 'python-fbas'], capture_output=True)
    except BaseException:
        pass  # Ignore errors if pkill fails


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark python-fbas vs Rust fbas_analyzer')
    parser.add_argument('--timeout', type=int, default=10,
                        help='Timeout in seconds per test (default: 10)')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='benchmark_results',
        help='Output directory for results (default: benchmark_results)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to test (default: all)')

    args = parser.parse_args()

    # Setup paths
    repo_root = Path(__file__).resolve().parent.parent
    test_data_dir = repo_root / "tests/test_data/random"
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / "benchmark" / output_dir

    # Create output directories
    output_dir.mkdir(exist_ok=True)

    print(f"🚀 FBAS Tools Benchmark")
    print(f"⏰ Timeout: {args.timeout} seconds per test")
    print(f"📁 Output directory: {output_dir}")

    # Check if tools are available
    rust_tool = repo_root / "fbas_analyzer/target/release/fbas_analyzer"
    if not rust_tool.exists():
        print("❌ Error: Rust fbas_analyzer not found. Please build it first:")
        print("   git clone https://github.com/trudi-group/fbas_analyzer.git")
        print("   cd fbas_analyzer && cargo build --release")
        sys.exit(1)
    rust_tool_path = shlex.quote(str(rust_tool))

    # Get all test files, excluding unwanted ones
    all_test_files = list(test_data_dir.glob("*.json"))
    test_files = [f for f in all_test_files if "for_stellar_core" not in f.name and not f.name.endswith(
        "_.json") and not f.name.endswith("_orgs.json")]
    if args.max_files:
        test_files = test_files[:args.max_files]

    print(f"📊 Testing {len(test_files)} files...")

    # Resolve python-fbas command
    python_fbas_path = shutil.which("python-fbas")
    if python_fbas_path:
        python_fbas_cmd = shlex.quote(python_fbas_path)
    else:
        python_fbas_cmd = f"{shlex.quote(sys.executable)} -m python_fbas.main"

    # Run benchmarks
    all_results = []

    for i, original_file in enumerate(test_files, 1):
        try:
            # Run benchmarks directly on original files
            print(f"[{i}/{len(test_files)}]", end=" ")
            file_results = run_benchmark_on_file(
                original_file,
                args.timeout,
                python_fbas_cmd,
                rust_tool_path)
            all_results.extend(file_results)

        except Exception as e:
            print(f"  ❌ Error processing {original_file.name}: {e}")
            continue

    # Generate reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    markdown_file = output_dir / f"benchmark_results_{timestamp}.md"
    csv_file = output_dir / f"benchmark_results_{timestamp}.csv"

    print(f"\n📝 Generating reports...")
    generate_markdown_report(all_results, args.timeout, markdown_file)
    generate_csv_report(all_results, csv_file)

    print(f"✅ Benchmark complete!")
    print(f"📄 Markdown report: {markdown_file}")
    print(f"📊 CSV data: {csv_file}")

    # Print quick summary
    total_tests = len(all_results)
    python_successes = sum(
        1 for r in all_results if r['python_status'] == 'success')
    rust_successes = sum(
        1 for r in all_results if r['rust_status'] == 'success')

    print(
        f"\n📈 Summary: {python_successes}/{total_tests} Python successful, {rust_successes}/{total_tests} Rust successful")

    # Clean up any remaining processes
    print("🧹 Cleaning up remaining processes...")
    cleanup_remaining_processes()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
        cleanup_remaining_processes()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        cleanup_remaining_processes()
        sys.exit(1)
