# Results Gzip-based Molecular Classification
|Data Set|       Task       |   Split    |AUROC/RMSE (Valid)|F1/MAE (Valid)|-/R (Valid)|AUROC/RMSE (Test)| F1/MAE (Test) |  -/R (Test)   |compressor|
|--------|------------------|------------|------------------|--------------|-----------|-----------------|---------------|---------------|----------|
|bbbp    |classification_vec|scaffold    |0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.677 +/- 0.017  |0.671 +/- 0.017|0.0 +/- 0.0    |Gzip      |
|tox21   |classification_vec|scaffold    |0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.602 +/- 0.02   |0.177 +/- 0.01 |0.0 +/- 0.0    |Gzip      |
|sider   |classification_vec|scaffold    |0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.566 +/- 0.007  |0.632 +/- 0.005|0.0 +/- 0.0    |Gzip      |
|clintox |classification_vec|scaffold    |0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.568 +/- 0.05   |0.741 +/- 0.016|0.0 +/- 0.0    |Gzip      |
|lipo    |regression        |scaffold_vec|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.985 +/- 0.021  |0.791 +/- 0.015|0.488 +/- 0.028|Gzip      |
|esol    |regression        |scaffold_vec|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.059 +/- 0.019  |0.812 +/- 0.02 |0.832 +/- 0.01 |Gzip      |
|freesolv|regression        |scaffold_vec|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.082 +/- 0.163  |2.241 +/- 0.158|0.64 +/- 0.053 |Gzip      |
