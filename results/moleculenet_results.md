# Results Gzip-based Molecular Classification

|Data Set|     Task     | Split  |AUROC/RMSE (Valid)|F1/MAE (Valid)|-/R (Valid)|AUROC/RMSE (Test)| F1/MAE (Test) |  -/R (Test)   |compressor|
|--------|--------------|--------|------------------|--------------|-----------|-----------------|---------------|---------------|----------|
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.642 +/- 0.025  |0.633 +/- 0.025|0.0 +/- 0.0    |Gzip      |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.59 +/- 0.02    |0.18 +/- 0.016 |0.0 +/- 0.0    |Gzip      |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.606 +/- 0.009  |0.671 +/- 0.011|0.0 +/- 0.0    |Gzip      |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.768 +/- 0.045  |0.916 +/- 0.018|0.0 +/- 0.0    |Gzip      |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.985 +/- 0.021  |0.791 +/- 0.015|0.488 +/- 0.028|Gzip      |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.059 +/- 0.019  |0.812 +/- 0.02 |0.832 +/- 0.01 |Gzip      |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.082 +/- 0.163  |2.241 +/- 0.158|0.64 +/- 0.053 |Gzip      |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.704 +/- 0.013  |0.88 +/- 0.003 |0.0 +/- 0.0    |Gzip      |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.042 +/- 0.0    |0.026 +/- 0.0  |0.916 +/- 0.002|Gzip      |
