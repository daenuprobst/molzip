# Results Gzip-based Molecular Classification
|Data Set|       Task       | Split  |AUROC/RMSE (Valid)|F1/MAE (Valid)|-/R (Valid)|AUROC/RMSE (Test)|F1/MAE (Test)| -/R (Test)  |compressor|
|--------|------------------|--------|------------------|--------------|-----------|-----------------|-------------|-------------|----------|
|bbbp    |classification_vec|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.686 +/- 0.0    |0.68 +/- 0.0 |0.0 +/- 0.0  |Gzip      |
|tox21   |classification_vec|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.591 +/- 0.0    |0.184 +/- 0.0|0.0 +/- 0.0  |Gzip      |
|sider   |classification_vec|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.581 +/- 0.0    |0.631 +/- 0.0|0.0 +/- 0.0  |Gzip      |
|clintox |classification_vec|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.598 +/- 0.0    |0.76 +/- 0.0 |0.0 +/- 0.0  |Gzip      |
|lipo    |regression_vec    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.912 +/- 0.0    |0.704 +/- 0.0|0.581 +/- 0.0|Gzip      |
|esol    |regression_vec    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.164 +/- 0.0    |0.847 +/- 0.0|0.804 +/- 0.0|Gzip      |
|freesolv|regression_vec    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.355 +/- 0.0    |2.628 +/- 0.0|0.74 +/- 0.0 |Gzip      |
|hiv     |classification_vec|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.716 +/- 0.0    |0.881 +/- 0.0|0.0 +/- 0.0  |Gzip      |
|qm8     |regression_vec    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.039 +/- 0.0    |0.022 +/- 0.0|0.924 +/- 0.0|Gzip      |
