# FBAS Tools Benchmark Results

**Generated:** 2025-07-16 05:12:56
**Timeout:** 10 seconds per test

## Commands Used

### Python-fbas
```bash
python-fbas --fbas=<file> check-intersection
python-fbas --fbas=<file> min-blocking-set
python-fbas --fbas=<file> min-splitting-set
```

### Rust fbas_analyzer
```bash
./fbas_analyzer <file> --alternative-quorum-intersection-check --results-only
./fbas_analyzer <file> -b --results-only
./fbas_analyzer <file> -s --results-only
```

## Summary

- **Total tests:** 204
- **Python successful:** 189/204
- **Rust successful:** 96/204
- **Both successful:** 96/204

## Results

| Handle | Nodes | Intersection |         | Blocking |         | Splitting |         |
|--------|-------|--------------|---------|----------|---------|-----------|---------|
|        |       | Python       | Rust    | Python   | Rust    | Python    | Rust    |
| N5F1   | 5     | 0.167s       | 0.003s  | 0.220s   | 0.004s  | 0.175s    | 0.033s  |
| N5F2   | 5     | 0.204s       | 0.003s  | 0.237s   | 0.006s  | 0.174s    | 0.124s  |
| N5F3   | 5     | 0.187s       | 0.003s  | 0.196s   | 0.004s  | 0.171s    | 0.025s  |
| N5F4   | 5     | 0.190s       | 0.002s  | 0.182s   | 0.002s  | 0.177s    | 0.002s  |
| N6F1   | 6     | 0.178s       | 0.004s  | 0.250s   | 0.006s  | 0.173s    | 0.153s  |
| N6F2   | 6     | 0.171s       | 0.003s  | 0.295s   | 0.022s  | 0.182s    | 7.644s  |
| N6F3   | 6     | 0.199s       | 0.003s  | 0.291s   | 0.012s  | 0.175s    | 1.257s  |
| N6F4   | 6     | 0.188s       | 0.002s  | 0.178s   | 0.003s  | 0.178s    | 0.003s  |
| N6F5   | 6     | 0.197s       | 0.003s  | 0.196s   | 0.004s  | 0.175s    | 0.032s  |
| N8F1   | 8     | 0.177s       | 0.006s  | 0.441s   | 0.240s  | 0.193s    | timeout |
| N8F2   | 8     | 0.173s       | 0.010s  | 0.492s   | 0.777s  | 0.238s    | timeout |
| N12F1  | 12    | 0.185s       | 0.076s  | 1.076s   | timeout | 1.556s    | timeout |
| N12F10 | 12    | 0.187s       | 0.005s  | 0.670s   | 1.464s  | 0.179s    | timeout |
| N12F11 | 12    | 0.177s       | 0.002s  | 0.529s   | 0.003s  | 0.178s    | 0.002s  |
| N12F2  | 12    | 0.180s       | 0.156s  | 1.569s   | timeout | 5.524s    | timeout |
| N12F3  | 12    | 0.188s       | 0.137s  | 1.605s   | timeout | 4.004s    | timeout |
| N12F4  | 12    | 0.197s       | 0.090s  | 1.512s   | timeout | 0.410s    | timeout |
| N12F5  | 12    | 0.190s       | 0.096s  | 1.712s   | timeout | 0.304s    | timeout |
| N12F6  | 12    | 0.197s       | 0.087s  | 1.574s   | timeout | 0.184s    | timeout |
| N12F7  | 12    | 0.178s       | 0.092s  | 1.496s   | timeout | 0.196s    | timeout |
| N12F8  | 12    | 0.187s       | 0.067s  | 1.506s   | timeout | 0.203s    | timeout |
| N12F9  | 12    | 0.177s       | 0.055s  | 1.480s   | timeout | 0.185s    | timeout |
| N13F1  | 13    | 0.179s       | timeout | 1.809s   | timeout | timeout   | timeout |
| N13F10 | 13    | 0.199s       | 0.002s  | 1.435s   | 0.003s  | 0.179s    | timeout |
| N13F11 | 13    | 0.203s       | 0.003s  | 0.289s   | 0.003s  | 0.179s    | 0.003s  |
| N13F12 | 13    | 0.191s       | 0.003s  | 1.023s   | 0.003s  | 0.180s    | 0.003s  |
| N13F2  | 13    | 0.207s       | 9.545s  | 1.718s   | timeout | 3.692s    | timeout |
| N13F3  | 13    | 0.205s       | 1.051s  | 1.972s   | timeout | timeout   | timeout |
| N13F4  | 13    | 0.197s       | 0.271s  | 2.006s   | timeout | timeout   | timeout |
| N13F5  | 13    | 0.188s       | 0.205s  | 1.900s   | timeout | 0.210s    | timeout |
| N13F6  | 13    | 0.195s       | 0.183s  | 1.949s   | timeout | 0.453s    | timeout |
| N13F7  | 13    | 0.197s       | 0.147s  | 1.983s   | timeout | 0.209s    | timeout |
| N13F8  | 13    | 0.192s       | 0.155s  | 1.952s   | timeout | 0.200s    | timeout |
| N13F9  | 13    | 0.181s       | 0.110s  | 1.963s   | timeout | 0.216s    | timeout |
| N14F1  | 14    | 0.197s       | 0.203s  | 7.884s   | timeout | timeout   | timeout |
| N14F10 | 14    | 0.179s       | 0.614s  | 2.304s   | timeout | 0.195s    | timeout |
| N14F11 | 14    | 0.181s       | 0.648s  | 2.248s   | timeout | 0.191s    | timeout |
| N14F12 | 14    | 0.186s       | 0.014s  | 0.816s   | 9.839s  | 0.178s    | timeout |
| N14F13 | 14    | 0.183s       | 0.005s  | 0.537s   | 0.389s  | 0.183s    | timeout |
| N14F2  | 14    | 0.195s       | 5.253s  | 2.233s   | timeout | 9.091s    | timeout |
| N14F3  | 14    | 0.186s       | 6.326s  | 2.413s   | timeout | 3.028s    | timeout |
| N14F4  | 14    | 0.209s       | 0.104s  | 2.687s   | timeout | 1.304s    | timeout |
| N14F5  | 14    | 0.199s       | 0.643s  | 2.482s   | timeout | 0.362s    | timeout |
| N14F6  | 14    | 0.189s       | 0.816s  | 2.326s   | timeout | 0.242s    | timeout |
| N14F7  | 14    | 0.201s       | 0.668s  | 2.447s   | timeout | 0.280s    | timeout |
| N14F8  | 14    | 0.192s       | 0.676s  | 2.444s   | timeout | 0.226s    | timeout |
| N14F9  | 14    | 0.194s       | 0.587s  | 2.397s   | timeout | 0.228s    | timeout |
| N16F1  | 16    | 0.196s       | timeout | timeout  | timeout | timeout   | timeout |
| N16F10 | 16    | 0.195s       | 3.324s  | 3.225s   | timeout | 0.202s    | timeout |
| N16F11 | 16    | 0.206s       | 1.271s  | 2.935s   | timeout | 0.196s    | timeout |
| N16F12 | 16    | 0.204s       | 3.214s  | 3.514s   | timeout | 0.203s    | timeout |
| N16F13 | 16    | 0.185s       | 4.408s  | 3.025s   | timeout | 0.196s    | timeout |
| N16F14 | 16    | 0.192s       | 0.003s  | 1.253s   | 0.003s  | 0.185s    | timeout |
| N16F15 | 16    | 0.199s       | 0.003s  | 0.188s   | 0.003s  | 0.194s    | 0.003s  |
| N16F2  | 16    | 0.199s       | timeout | timeout  | timeout | timeout   | timeout |
| N16F3  | 16    | 0.208s       | timeout | 3.343s   | timeout | timeout   | timeout |
| N16F4  | 16    | 0.187s       | 6.882s  | 3.602s   | timeout | timeout   | timeout |
| N16F5  | 16    | 0.193s       | 7.683s  | 3.494s   | timeout | 2.894s    | timeout |
| N16F6  | 16    | 0.205s       | 5.972s  | 3.408s   | timeout | 2.116s    | timeout |
| N16F7  | 16    | 0.192s       | 5.810s  | 3.414s   | timeout | 0.209s    | timeout |
| N16F8  | 16    | 0.181s       | 5.539s  | 3.540s   | timeout | 0.845s    | timeout |
| N16F9  | 16    | 0.186s       | 4.817s  | 3.191s   | timeout | 0.208s    | timeout |
| N8     | 24    | 0.198s       | 0.015s  | 0.447s   | 0.510s  | 0.257s    | timeout |
| N10    | 30    | 0.191s       | 0.471s  | 0.970s   | timeout | 0.601s    | timeout |
| N12    | 36    | 0.200s       | 4.685s  | 1.037s   | timeout | 0.957s    | timeout |
| N13    | 39    | 0.201s       | 0.492s  | timeout  | timeout | timeout   | timeout |
| N16    | 48    | 0.188s       | timeout | 2.836s   | timeout | timeout   | timeout |
| N24    | 72    | 0.200s       | timeout | timeout  | timeout | timeout   | timeout |

## File Handle Mapping

| Handle | File Name                                                   |
|--------|-------------------------------------------------------------|
| N10    | almost_symmetric_network_10_orgs.json                       |
| N12    | almost_symmetric_network_12_orgs.json                       |
| N12F1  | almost_symmetric_network_12_orgs_delete_prob_factor_1.json  |
| N12F10 | almost_symmetric_network_12_orgs_delete_prob_factor_10.json |
| N12F11 | almost_symmetric_network_12_orgs_delete_prob_factor_11.json |
| N12F2  | almost_symmetric_network_12_orgs_delete_prob_factor_2.json  |
| N12F3  | almost_symmetric_network_12_orgs_delete_prob_factor_3.json  |
| N12F4  | almost_symmetric_network_12_orgs_delete_prob_factor_4.json  |
| N12F5  | almost_symmetric_network_12_orgs_delete_prob_factor_5.json  |
| N12F6  | almost_symmetric_network_12_orgs_delete_prob_factor_6.json  |
| N12F7  | almost_symmetric_network_12_orgs_delete_prob_factor_7.json  |
| N12F8  | almost_symmetric_network_12_orgs_delete_prob_factor_8.json  |
| N12F9  | almost_symmetric_network_12_orgs_delete_prob_factor_9.json  |
| N13    | almost_symmetric_network_13_orgs.json                       |
| N13F1  | almost_symmetric_network_13_orgs_delete_prob_factor_1.json  |
| N13F10 | almost_symmetric_network_13_orgs_delete_prob_factor_10.json |
| N13F11 | almost_symmetric_network_13_orgs_delete_prob_factor_11.json |
| N13F12 | almost_symmetric_network_13_orgs_delete_prob_factor_12.json |
| N13F2  | almost_symmetric_network_13_orgs_delete_prob_factor_2.json  |
| N13F3  | almost_symmetric_network_13_orgs_delete_prob_factor_3.json  |
| N13F4  | almost_symmetric_network_13_orgs_delete_prob_factor_4.json  |
| N13F5  | almost_symmetric_network_13_orgs_delete_prob_factor_5.json  |
| N13F6  | almost_symmetric_network_13_orgs_delete_prob_factor_6.json  |
| N13F7  | almost_symmetric_network_13_orgs_delete_prob_factor_7.json  |
| N13F8  | almost_symmetric_network_13_orgs_delete_prob_factor_8.json  |
| N13F9  | almost_symmetric_network_13_orgs_delete_prob_factor_9.json  |
| N14F1  | almost_symmetric_network_14_orgs_delete_prob_factor_1.json  |
| N14F10 | almost_symmetric_network_14_orgs_delete_prob_factor_10.json |
| N14F11 | almost_symmetric_network_14_orgs_delete_prob_factor_11.json |
| N14F12 | almost_symmetric_network_14_orgs_delete_prob_factor_12.json |
| N14F13 | almost_symmetric_network_14_orgs_delete_prob_factor_13.json |
| N14F2  | almost_symmetric_network_14_orgs_delete_prob_factor_2.json  |
| N14F3  | almost_symmetric_network_14_orgs_delete_prob_factor_3.json  |
| N14F4  | almost_symmetric_network_14_orgs_delete_prob_factor_4.json  |
| N14F5  | almost_symmetric_network_14_orgs_delete_prob_factor_5.json  |
| N14F6  | almost_symmetric_network_14_orgs_delete_prob_factor_6.json  |
| N14F7  | almost_symmetric_network_14_orgs_delete_prob_factor_7.json  |
| N14F8  | almost_symmetric_network_14_orgs_delete_prob_factor_8.json  |
| N14F9  | almost_symmetric_network_14_orgs_delete_prob_factor_9.json  |
| N16    | almost_symmetric_network_16_orgs.json                       |
| N16F1  | almost_symmetric_network_16_orgs_delete_prob_factor_1.json  |
| N16F10 | almost_symmetric_network_16_orgs_delete_prob_factor_10.json |
| N16F11 | almost_symmetric_network_16_orgs_delete_prob_factor_11.json |
| N16F12 | almost_symmetric_network_16_orgs_delete_prob_factor_12.json |
| N16F13 | almost_symmetric_network_16_orgs_delete_prob_factor_13.json |
| N16F14 | almost_symmetric_network_16_orgs_delete_prob_factor_14.json |
| N16F15 | almost_symmetric_network_16_orgs_delete_prob_factor_15.json |
| N16F2  | almost_symmetric_network_16_orgs_delete_prob_factor_2.json  |
| N16F3  | almost_symmetric_network_16_orgs_delete_prob_factor_3.json  |
| N16F4  | almost_symmetric_network_16_orgs_delete_prob_factor_4.json  |
| N16F5  | almost_symmetric_network_16_orgs_delete_prob_factor_5.json  |
| N16F6  | almost_symmetric_network_16_orgs_delete_prob_factor_6.json  |
| N16F7  | almost_symmetric_network_16_orgs_delete_prob_factor_7.json  |
| N16F8  | almost_symmetric_network_16_orgs_delete_prob_factor_8.json  |
| N16F9  | almost_symmetric_network_16_orgs_delete_prob_factor_9.json  |
| N24    | almost_symmetric_network_24_orgs.json                       |
| N5F1   | almost_symmetric_network_5_orgs_delete_prob_factor_1.json   |
| N5F2   | almost_symmetric_network_5_orgs_delete_prob_factor_2.json   |
| N5F3   | almost_symmetric_network_5_orgs_delete_prob_factor_3.json   |
| N5F4   | almost_symmetric_network_5_orgs_delete_prob_factor_4.json   |
| N6F1   | almost_symmetric_network_6_orgs_delete_prob_factor_1.json   |
| N6F2   | almost_symmetric_network_6_orgs_delete_prob_factor_2.json   |
| N6F3   | almost_symmetric_network_6_orgs_delete_prob_factor_3.json   |
| N6F4   | almost_symmetric_network_6_orgs_delete_prob_factor_4.json   |
| N6F5   | almost_symmetric_network_6_orgs_delete_prob_factor_5.json   |
| N8     | almost_symmetric_network_8_orgs.json                        |
| N8F1   | almost_symmetric_network_8_orgs_delete_prob_factor_1.json   |
| N8F2   | almost_symmetric_network_8_orgs_delete_prob_factor_2.json   |
