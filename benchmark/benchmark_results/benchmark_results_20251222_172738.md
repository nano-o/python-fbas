# FBAS Tools Benchmark Results

**Generated:** 2025-12-22 17:27:38
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

**Note:** Test files are automatically converted from python-fbas format to stellarbeat format for fbas_analyzer.

## Summary

- **Total tests:** 186
- **Python successful:** 174/186
- **Rust successful:** 91/186
- **Both successful:** 91/186

## Results

| Handle | Orgs | Intersection |         | Blocking |         | Splitting |         |
|--------:------:--------------:---------:----------:---------:-----------:---------|
|        |      | Python       | Rust    | Python   | Rust    | Python    | Rust    |
| N5F1   | 5    | 0.210s       | 0.004s  | 0.293s   | 0.004s  | 0.207s    | 0.029s  |
| N5F2   | 5    | 0.219s       | 0.004s  | 0.320s   | 0.006s  | 0.218s    | 0.131s  |
| N5F3   | 5    | 0.209s       | 0.003s  | 0.241s   | 0.004s  | 0.200s    | 0.027s  |
| N5F4   | 5    | 0.230s       | 0.004s  | 0.231s   | 0.003s  | 0.227s    | 0.004s  |
| N6F1   | 6    | 0.213s       | 0.004s  | 0.343s   | 0.006s  | 0.212s    | 0.163s  |
| N6F2   | 6    | 0.223s       | 0.004s  | 0.446s   | 0.024s  | 0.229s    | 8.007s  |
| N6F3   | 6    | 0.218s       | 0.004s  | 0.405s   | 0.012s  | 0.205s    | 1.304s  |
| N6F4   | 6    | 0.207s       | 0.003s  | 0.211s   | 0.003s  | 0.206s    | 0.004s  |
| N6F5   | 6    | 0.221s       | 0.003s  | 0.264s   | 0.004s  | 0.224s    | 0.030s  |
| N8F1   | 8    | 0.202s       | 0.008s  | 0.629s   | 0.252s  | 0.235s    | timeout |
| N8F2   | 8    | 0.212s       | 0.012s  | 0.738s   | 0.799s  | 0.238s    | timeout |
| N12F1  | 12   | 0.221s       | 0.068s  | 1.547s   | timeout | 2.519s    | timeout |
| N12F10 | 12   | 0.220s       | 0.007s  | 0.970s   | 2.021s  | 0.213s    | timeout |
| N12F11 | 12   | 0.221s       | 0.003s  | 0.751s   | 0.004s  | 0.214s    | 0.003s  |
| N12F2  | 12   | 0.209s       | 0.112s  | 2.202s   | timeout | 3.898s    | timeout |
| N12F3  | 12   | 0.236s       | 0.124s  | 2.270s   | timeout | timeout   | timeout |
| N12F4  | 12   | 0.221s       | 0.093s  | 2.062s   | timeout | 0.581s    | timeout |
| N12F5  | 12   | 0.225s       | 0.127s  | 2.202s   | timeout | 0.318s    | timeout |
| N12F6  | 12   | 0.226s       | 0.091s  | 2.291s   | timeout | 0.229s    | timeout |
| N12F7  | 12   | 0.214s       | 0.083s  | 2.206s   | timeout | 0.249s    | timeout |
| N12F8  | 12   | 0.217s       | 0.088s  | 2.111s   | timeout | 0.238s    | timeout |
| N12F9  | 12   | 0.207s       | 0.051s  | 2.093s   | timeout | 0.221s    | timeout |
| N13F1  | 13   | 0.209s       | timeout | 2.366s   | timeout | timeout   | timeout |
| N13F10 | 13   | 0.238s       | 0.003s  | 2.127s   | 0.003s  | 0.208s    | timeout |
| N13F11 | 13   | 0.219s       | 0.003s  | 0.392s   | 0.004s  | 0.208s    | 0.004s  |
| N13F12 | 13   | 0.207s       | 0.003s  | 1.651s   | 0.003s  | 0.211s    | 0.003s  |
| N13F2  | 13   | 0.221s       | 7.974s  | 2.496s   | timeout | 2.693s    | timeout |
| N13F3  | 13   | 0.232s       | 1.146s  | 2.908s   | timeout | timeout   | timeout |
| N13F4  | 13   | 0.224s       | 0.270s  | 2.747s   | timeout | timeout   | timeout |
| N13F5  | 13   | 0.215s       | 0.189s  | 2.821s   | timeout | 0.241s    | timeout |
| N13F6  | 13   | 0.222s       | 0.173s  | 2.989s   | timeout | 0.610s    | timeout |
| N13F7  | 13   | 0.214s       | 0.147s  | 2.751s   | timeout | 0.270s    | timeout |
| N13F8  | 13   | 0.235s       | 0.131s  | 2.733s   | timeout | 0.225s    | timeout |
| N13F9  | 13   | 0.217s       | 0.141s  | 2.713s   | timeout | 0.232s    | timeout |
| N14F1  | 14   | 0.210s       | 0.217s  | 6.749s   | timeout | timeout   | timeout |
| N14F10 | 14   | 0.203s       | 0.475s  | 3.214s   | timeout | 0.240s    | timeout |
| N14F11 | 14   | 0.208s       | 0.428s  | 3.236s   | timeout | 0.236s    | timeout |
| N14F12 | 14   | 0.228s       | 0.011s  | 1.238s   | 6.679s  | 0.212s    | timeout |
| N14F13 | 14   | 0.211s       | 0.005s  | 0.748s   | 0.363s  | 0.209s    | timeout |
| N14F2  | 14   | 0.213s       | 6.143s  | 3.315s   | timeout | timeout   | timeout |
| N14F3  | 14   | 0.218s       | 7.097s  | 3.434s   | timeout | 7.599s    | timeout |
| N14F4  | 14   | 0.220s       | 0.114s  | 3.533s   | timeout | 0.855s    | timeout |
| N14F5  | 14   | 0.223s       | 0.698s  | 3.565s   | timeout | 0.457s    | timeout |
| N14F6  | 14   | 0.218s       | 0.805s  | 3.749s   | timeout | 0.292s    | timeout |
| N14F7  | 14   | 0.226s       | 0.677s  | 3.353s   | timeout | 0.457s    | timeout |
| N14F8  | 14   | 0.223s       | 0.705s  | 3.382s   | timeout | 0.253s    | timeout |
| N14F9  | 14   | 0.233s       | 0.689s  | 3.270s   | timeout | 0.254s    | timeout |
| N16F1  | 16   | 0.220s       | timeout | timeout  | timeout | timeout   | timeout |
| N16F10 | 16   | 0.226s       | 3.656s  | 5.089s   | timeout | 0.230s    | timeout |
| N16F11 | 16   | 0.227s       | 0.976s  | 3.969s   | timeout | 0.240s    | timeout |
| N16F12 | 16   | 0.222s       | 2.913s  | 5.539s   | timeout | 0.245s    | timeout |
| N16F13 | 16   | 0.221s       | 2.596s  | 4.791s   | timeout | 0.220s    | timeout |
| N16F14 | 16   | 0.232s       | 0.003s  | 1.964s   | 0.003s  | 0.215s    | timeout |
| N16F15 | 16   | 0.214s       | 0.003s  | 0.221s   | 0.004s  | 0.205s    | 0.004s  |
| N16F2  | 16   | 0.222s       | timeout | timeout  | timeout | timeout   | timeout |
| N16F3  | 16   | 0.222s       | timeout | 4.993s   | timeout | timeout   | timeout |
| N16F4  | 16   | 0.218s       | 7.052s  | 5.512s   | timeout | timeout   | timeout |
| N16F5  | 16   | 0.218s       | 9.175s  | 5.164s   | timeout | 1.744s    | timeout |
| N16F6  | 16   | 0.229s       | 6.293s  | 5.297s   | timeout | 1.973s    | timeout |
| N16F7  | 16   | 0.215s       | 5.798s  | 5.163s   | timeout | 0.229s    | timeout |
| N16F8  | 16   | 0.208s       | 5.582s  | 5.801s   | timeout | 0.962s    | timeout |
| N16F9  | 16   | 0.223s       | 5.266s  | 5.075s   | timeout | 0.301s    | timeout |

## File Handle Mapping

| Handle | File Name                                                   |
|--------|-------------------------------------------------------------|
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
| N5F1   | almost_symmetric_network_5_orgs_delete_prob_factor_1.json   |
| N5F2   | almost_symmetric_network_5_orgs_delete_prob_factor_2.json   |
| N5F3   | almost_symmetric_network_5_orgs_delete_prob_factor_3.json   |
| N5F4   | almost_symmetric_network_5_orgs_delete_prob_factor_4.json   |
| N6F1   | almost_symmetric_network_6_orgs_delete_prob_factor_1.json   |
| N6F2   | almost_symmetric_network_6_orgs_delete_prob_factor_2.json   |
| N6F3   | almost_symmetric_network_6_orgs_delete_prob_factor_3.json   |
| N6F4   | almost_symmetric_network_6_orgs_delete_prob_factor_4.json   |
| N6F5   | almost_symmetric_network_6_orgs_delete_prob_factor_5.json   |
| N8F1   | almost_symmetric_network_8_orgs_delete_prob_factor_1.json   |
| N8F2   | almost_symmetric_network_8_orgs_delete_prob_factor_2.json   |
