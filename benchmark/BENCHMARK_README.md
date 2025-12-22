# FBAS Tools Benchmark

This directory contains a benchmark script for comparing the performance of `python-fbas` against the Rust `fbas_analyzer` tool.

## Prerequisites

1. **Python-fbas**: Ensure `python-fbas` is installed and available in PATH (the script falls back to `python -m python_fbas.main` if needed)
2. **Rust fbas_analyzer**: Clone and build the Rust tool:
   ```bash
   git clone https://github.com/trudi-group/fbas_analyzer.git
   cd fbas_analyzer
   cargo build --release
   cd ..
   ```

## Usage

### Basic Usage

Run benchmarks on all available test files with default 10-second timeout:

```bash
python3 benchmark/benchmark_fbas_tools.py
```

Results are written to `benchmark/benchmark_results/` by default.

### Custom Options

```bash
# Custom timeout (20 seconds)
python3 benchmark/benchmark_fbas_tools.py --timeout 20

# Limit number of test files (useful for quick testing)
python3 benchmark/benchmark_fbas_tools.py --max-files 10

# Custom output directory (relative to benchmark/)
python3 benchmark/benchmark_fbas_tools.py --output-dir my_benchmark_results

# Combination of options
python3 benchmark/benchmark_fbas_tools.py --timeout 30 --max-files 5 --output-dir quick_test
```


## Test Coverage

The benchmark tests three types of FBAS analysis:

1. **Intersection Check**: Tests if all quorums intersect
2. **Minimal Blocking Sets**: Finds minimal sets that can block consensus
3. **Minimal Splitting Sets**: Finds minimal sets that can split consensus

## Test Data

Uses all JSON files from `tests/test_data/random/` directory, which includes:
- Networks of various sizes (6-72 nodes)
- Different organizational structures
- Multiple network degradation scenarios

The benchmark script auto-detects the input format (stellarbeat or python-fbas)
and converts to stellarbeat format for the Rust `fbas_analyzer` tool when needed.

## Output

The benchmark generates two types of output:

### 1. Markdown Report (`benchmark_results_TIMESTAMP.md`)

Contains:
- Commands used for both tools
- Summary statistics
- Results table with execution times
- Clear formatting for documentation

### 2. CSV Data (`benchmark_results_TIMESTAMP.csv`)

Contains:
- Detailed results for each test
- Machine-readable format for analysis
- Fields: file_name, network_size, test_type, python_time, python_status, rust_time, rust_status, speedup

## Result Interpretation

### Status Values
- `success`: Test completed successfully
- `timeout`: Test exceeded the timeout limit
- `error`: Test failed with an error

### Timing
- Times are reported in seconds (e.g., `0.123s`)
- `timeout` indicates the test did not complete within the specified time limit
- Missing times indicate unsuccessful tests

## Example Output

```
🚀 FBAS Tools Benchmark
⏰ Timeout: 10 seconds per test
📁 Output directory: benchmark_results
📊 Testing 211 files...
[1/211]   Testing almost_symmetric_network_10_orgs.json... 5/6 completed
[2/211]   Testing almost_symmetric_network_12_orgs.json... 4/6 completed
...
📝 Generating reports...
✅ Benchmark complete!
📄 Markdown report: benchmark_results/benchmark_results_20231201_143022.md
📊 CSV data: benchmark_results/benchmark_results_20231201_143022.csv

📈 Summary: 387/633 Python successful, 298/633 Rust successful
```
