# Fine-Grained GPU×EMC Profiling Report

**Generated**: 2026-05-22 18:18:44
**Model**: Phi-3-mini-Q4
**GPU levels**: 11 (306-1300 MHz, 11 steps)
**EMC levels**: 1 (204-3199 MHz)
**CPU**: 1036 MHz (fixed)
**Phases**: decode, mixed
**Total valid runs**: 1320

## GPU Frequency Effect (averaged across EMC)

| GPU (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) | tok/J |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 306 | 9.4320 | 324.4 | 3.6 | 30.2 | 0.12 |
| 408 | 9.6628 | 265.5 | 4.0 | 38.5 | 0.10 |
| 510 | 8.8312 | 215.5 | 4.8 | 43.2 | 0.11 |
| 612 | 4.8103 | 113.8 | 9.5 | 42.8 | 0.22 |
| 714 | 4.1169 | 92.3 | 10.8 | 44.7 | 0.24 |
| 816 | 4.0359 | 86.2 | 11.5 | 46.6 | 0.25 |
| 918 | 4.0036 | 82.7 | 12.0 | 47.8 | 0.25 |
| 1020 | 3.9800 | 80.6 | 12.3 | 48.6 | 0.25 |
| 1122 | 3.9374 | 79.3 | 12.5 | 49.6 | 0.25 |
| 1224 | 3.9373 | 78.4 | 12.7 | 49.8 | 0.25 |
| 1300 | 3.9346 | 77.9 | 12.8 | 50.0 | 0.25 |

## EMC Frequency Effect (averaged across GPU)

| EMC (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) | tok/J |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 3199 | 5.5165 | 136.0 | 9.7 | 44.7 | 0.21 |

## Best E/tok Config per Workload

| Workload | Best Config | E/tok (J) | TPOT (ms) |
|:---:|:---:|:---:|:---:|
| p1024_o128 | GPU1020_EMC3199 | 3.7986 | 79.6 |
| p1024_o512 | GPU1122_EMC3199 | 3.5560 | 81.2 |
| p128_o256 | GPU1300_EMC3199 | 3.8642 | 77.3 |
| p128_o64 | GPU1020_EMC3199 | 3.7382 | 79.3 |
| p512_o128 | GPU1020_EMC3199 | 3.8076 | 79.6 |