
============================================================
FLASH TIMING STATISTICS
============================================================

--- INFERENCE ---
Total time: 85.6839s
Mean: 0.069100s ± 0.003574s
Median: 0.067712s
Min: 0.064782s
Max: 0.090005s
Variance: 0.000012775s²
95th percentile: 0.076223s
Throughput: 14.47 iterations/sec

--- DATA_TRANSFER ---
Total time: 2.5261s
Mean: 0.002037s ± 0.000546s
Median: 0.002183s
Min: 0.000328s
Max: 0.003688s
Variance: 0.000000298s²
95th percentile: 0.002538s
Throughput: 490.87 iterations/sec

--- UNET_CONDITION ---
Total time: 42.5415s
Mean: 0.034308s ± 0.002117s
Median: 0.034131s
Min: 0.030431s
Max: 0.051449s
Variance: 0.000004484s²
95th percentile: 0.036910s
Throughput: 29.15 iterations/sec

--- TOTAL_ITERATION ---
Total time: 45.0690s
Mean: 0.036346s ± 0.001989s
Median: 0.036211s
Min: 0.032006s
Max: 0.052862s
Variance: 0.000003957s²
95th percentile: 0.038441s
Throughput: 27.51 iterations/sec

--- OVERALL TRAINING ---
Total training time: 45.07s (0.75 minutes)
Average iteration time: 0.0363s
Estimated throughput: 27.51 iterations/second

--- TIME DISTRIBUTION ---
data_transfer: 5.6% (2.53s)
unet_condition: 94.4% (42.54s)

--- GPU MEMORY USAGE ---
Max memory allocated: 1.88 GB
Max memory reserved: 21.80 GB


============================================================
NO FLASH TIMING STATISTICS
============================================================

--- INFERENCE ---
Total time: 84.0060s
Mean: 0.067747s ± 0.001573s
Median: 0.067316s
Min: 0.065171s
Max: 0.080422s
Variance: 0.000002473s²
95th percentile: 0.070826s
Throughput: 14.76 iterations/sec

--- DATA_TRANSFER ---
Total time: 3.0545s
Mean: 0.002463s ± 0.000167s
Median: 0.002413s
Min: 0.001002s
Max: 0.003939s
Variance: 0.000000028s²
95th percentile: 0.002773s
Throughput: 405.96 iterations/sec

--- UNET_CONDITION ---
Total time: 41.4104s
Mean: 0.033396s ± 0.011296s
Median: 0.032946s
Min: 0.029679s
Max: 0.428914s
Variance: 0.000127598s²
95th percentile: 0.034960s
Throughput: 29.94 iterations/sec

--- TOTAL_ITERATION ---
Total time: 44.4662s
Mean: 0.035860s ± 0.011290s
Median: 0.035402s
Min: 0.032131s
Max: 0.431213s
Variance: 0.000127456s²
95th percentile: 0.037425s
Throughput: 27.89 iterations/sec

--- OVERALL TRAINING ---
Total training time: 44.47s (0.74 minutes)
Average iteration time: 0.0359s
Estimated throughput: 27.89 iterations/second

--- TIME DISTRIBUTION ---
data_transfer: 6.9% (3.05s)
unet_condition: 93.1% (41.41s)

--- GPU MEMORY USAGE ---
Max memory allocated: 1.87 GB
Max memory reserved: 21.83 GB