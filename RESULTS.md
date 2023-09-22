# Results Gzip-based Molecular Classification

|     Data Set      |     Task     | Split  |AUROC/RMSE (Valid)|F1/MAE (Valid) |AUROC/RMSE (Test)| F1/MAE (Test) |
|-------------------|--------------|--------|------------------|---------------|-----------------|---------------|
|sampl              |regression    |random  |3.119 +/- 0.353   |2.326 +/- 0.353|2.597 +/- 0.353  |2.035 +/- 0.353|
|sampl              |regression_vec|random  |1.994 +/- 0.83    |1.159 +/- 0.83 |2.176 +/- 0.83   |1.286 +/- 0.83 |
|delaney            |regression    |random  |1.451 +/- 0.019   |1.088 +/- 0.019|1.446 +/- 0.019  |1.097 +/- 0.019|
|delaney            |regression_vec|random  |1.088 +/- 0.012   |0.813 +/- 0.012|1.172 +/- 0.012  |0.878 +/- 0.012|
|lipo               |regression    |random  |1.007 +/- 0.048   |0.796 +/- 0.048|1.017 +/- 0.048  |0.805 +/- 0.048|
|lipo               |regression_vec|random  |0.938 +/- 0.03    |0.738 +/- 0.03 |0.925 +/- 0.03   |0.729 +/- 0.03 |
|sider              |classification|scaffold|0.551 +/- 0.0     |0.707 +/- 0.0  |0.577 +/- 0.0    |0.666 +/- 0.0  |
|sider              |classification|random  |0.587 +/- 0.009   |0.649 +/- 0.009|0.586 +/- 0.009  |0.655 +/- 0.009|
|bbbp               |classification|scaffold|0.931 +/- 0.0     |0.931 +/- 0.0  |0.639 +/- 0.0    |0.627 +/- 0.0  |
|bace_classification|classification|scaffold|0.694 +/- 0.0     |0.702 +/- 0.0  |0.701 +/- 0.0    |0.697 +/- 0.0  |
|bace_classification|classification|random  |0.796 +/- 0.009   |0.798 +/- 0.009|0.822 +/- 0.009  |0.819 +/- 0.009|
|clintox            |classification|scaffold|0.805 +/- 0.0     |0.854 +/- 0.0  |0.891 +/- 0.0    |0.891 +/- 0.0  |
|clintox            |classification|random  |0.865 +/- 0.093   |0.912 +/- 0.093|0.883 +/- 0.093  |0.896 +/- 0.093|
|tox21              |classification|scaffold|0.635 +/- 0.0     |0.247 +/- 0.0  |0.618 +/- 0.0    |0.227 +/- 0.0  |
|tox21              |classification|random  |0.685 +/- 0.009   |0.287 +/- 0.009|0.678 +/- 0.009  |0.282 +/- 0.009|
|hiv                |classification|scaffold|0.714 +/- 0.0     |0.901 +/- 0.0  |0.689 +/- 0.0    |0.887 +/- 0.0  |
|muv                |classification|random  |0.131 +/- 0.227   |0.033 +/- 0.227|0.137 +/- 0.227  |0.034 +/- 0.227|
|schneider          |classification|random  |0.0 +/- 0.0       |0.801 +/- 0.0  |0.0 +/- 0.0      |0.801 +/- 0.0  |

# Results Gzip-based QMOF regression

|Data Set|Split |AUROC/RMSE (Valid)|F1/MAE (Valid)|AUROC/RMSE (Test)|F1/MAE (Test) |
|--------|------|------------------|--------------|-----------------|--------------|
|MOF     |random|0.769 +/- 0.01    |0.594 +/- 0.01|0.782 +/- 0.01   |0.604 +/- 0.01|
