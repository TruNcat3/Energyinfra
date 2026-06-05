# Fine-Grained GPU×EMC Profiling Report

**Generated**: 2026-05-22 13:11:19
**Model**: Phi-3-mini-Q4
**GPU levels**: 11 (306-1300 MHz, 11 steps)
**EMC levels**: 1 (204-3199 MHz)
**CPU**: 1036 MHz (fixed)
**Phases**: decode, mixed
**Total valid runs**: 330

## GPU Frequency Effect (averaged across EMC)

| GPU (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) | tok/J |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 306 | 12.7040 | 478.4 | 2.1 | 27.3 | 0.08 |
| 408 | 11.2862 | 378.5 | 2.7 | 30.7 | 0.09 |
| 510 | 9.9790 | 288.6 | 3.5 | 35.3 | 0.10 |
| 612 | 9.5826 | 259.4 | 3.9 | 38.0 | 0.11 |
| 714 | 9.2124 | 235.0 | 4.4 | 40.5 | 0.11 |
| 816 | 9.1436 | 232.7 | 4.6 | 41.3 | 0.11 |
| 918 | 8.7174 | 204.4 | 5.0 | 43.7 | 0.12 |
| 1020 | 8.6116 | 194.6 | 5.2 | 45.3 | 0.12 |
| 1122 | 8.1911 | 191.7 | 5.3 | 45.9 | 0.12 |
| 1224 | 8.4350 | 189.4 | 5.4 | 46.2 | 0.12 |
| 1300 | 8.3716 | 183.8 | 5.5 | 46.6 | 0.12 |

## EMC Frequency Effect (averaged across GPU)

| EMC (MHz) | E/tok (J) | TPOT (ms) | TPS | Power (W) | tok/J |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 3199 | 9.4759 | 257.9 | 4.3 | 40.1 | 0.11 |

## Best E/tok Config per Workload

| Workload | Best Config | E/tok (J) | TPOT (ms) |
|:---:|:---:|:---:|:---:|
| p1024_o128 | GPU1122_EMC3199 | 6.4915 | 176.9 |
| p1024_o512 | GPU1122_EMC3199 | 7.6172 | 176.9 |
| p128_o256 | GPU1224_EMC3199 | 7.4955 | 157.4 |
| p128_o64 | GPU1122_EMC3199 | 6.0125 | 167.6 |
| p512_o128 | GPU1122_EMC3199 | 7.7526 | 172.7 |