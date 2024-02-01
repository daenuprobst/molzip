# Results Gzip-based Molecular Classification
|     Data Set      |       Task       | Split  |AUROC/RMSE (Valid)|F1/MAE (Valid)| -/R (Valid) |AUROC/RMSE (Test)|F1/MAE (Test) | -/R (Test)  |
|-------------------|------------------|--------|------------------|--------------|-------------|-----------------|--------------|-------------|
|bace_classification|classification    |scaffold|0.656 +/- 0.0     |0.662 +/- 0.0 |0.0 +/- 0.0  |0.668 +/- 0.0    |0.664 +/- 0.0 |0.0 +/- 0.0  |
|bace_classification|classification_vec|scaffold|0.663 +/- 0.0     |0.675 +/- 0.0 |0.0 +/- 0.0  |0.669 +/- 0.0    |0.645 +/- 0.0 |0.0 +/- 0.0  |
|bbbp               |classification    |scaffold|0.933 +/- 0.0     |0.936 +/- 0.0 |0.0 +/- 0.0  |0.665 +/- 0.0    |0.657 +/- 0.0 |0.0 +/- 0.0  |
|bbbp               |classification_vec|scaffold|0.898 +/- 0.0     |0.897 +/- 0.0 |0.0 +/- 0.0  |0.692 +/- 0.0    |0.686 +/- 0.0 |0.0 +/- 0.0  |
|clintox            |classification    |scaffold|0.947 +/- 0.0     |0.899 +/- 0.0 |0.0 +/- 0.0  |0.862 +/- 0.0    |0.838 +/- 0.0 |0.0 +/- 0.0  |
|clintox            |classification_vec|scaffold|0.599 +/- 0.0     |0.689 +/- 0.0 |0.0 +/- 0.0  |0.5 +/- 0.0      |0.548 +/- 0.0 |0.0 +/- 0.0  |
|tox21              |classification    |scaffold|0.747 +/- 0.0     |0.959 +/- 0.0 |0.0 +/- 0.0  |0.692 +/- 0.0    |0.958 +/- 0.0 |0.0 +/- 0.0  |
|tox21              |classification_vec|scaffold|0.664 +/- 0.0     |0.969 +/- 0.0 |0.0 +/- 0.0  |0.662 +/- 0.0    |0.968 +/- 0.0 |0.0 +/- 0.0  |
|delaney            |regression        |scaffold|1.419 +/- 0.0     |1.121 +/- 0.0 |0.705 +/- 0.0|1.51 +/- 0.0     |1.191 +/- 0.0 |0.704 +/- 0.0|
|delaney            |regression_vec    |scaffold|1.282 +/- 0.0     |1.02 +/- 0.0  |0.79 +/- 0.0 |1.271 +/- 0.0    |0.959 +/- 0.0 |0.838 +/- 0.0|
|bace_regression    |regression        |scaffold|0.735 +/- 0.0     |0.58 +/- 0.0  |0.382 +/- 0.0|1.174 +/- 0.0    |0.939 +/- 0.0 |0.739 +/- 0.0|
|bace_regression    |regression_vec    |scaffold|0.779 +/- 0.0     |0.621 +/- 0.0 |0.335 +/- 0.0|1.133 +/- 0.0    |0.892 +/- 0.0 |0.763 +/- 0.0|
|lipo               |regression        |scaffold|1.075 +/- 0.0     |0.846 +/- 0.0 |0.493 +/- 0.0|1.042 +/- 0.0    |0.824 +/- 0.0 |0.41 +/- 0.0 |
|lipo               |regression_vec    |scaffold|1.01 +/- 0.0      |0.781 +/- 0.0 |0.573 +/- 0.0|0.915 +/- 0.0    |0.704 +/- 0.0 |0.582 +/- 0.0|
|clearance          |regression        |scaffold|49.59 +/- 0.0     |38.781 +/- 0.0|0.429 +/- 0.0|49.885 +/- 0.0   |38.724 +/- 0.0|0.443 +/- 0.0|
|clearance          |regression_vec    |scaffold|48.037 +/- 0.0    |38.971 +/- 0.0|0.474 +/- 0.0|51.756 +/- 0.0   |40.683 +/- 0.0|0.392 +/- 0.0|
|hiv                |regression        |scaffold|0.138 +/- 0.0     |0.036 +/- 0.0 |0.349 +/- 0.0|0.17 +/- 0.0     |0.048 +/- 0.0 |0.342 +/- 0.0|
|hiv                |regression_vec    |scaffold|0.132 +/- 0.0     |0.034 +/- 0.0 |0.261 +/- 0.0|0.175 +/- 0.0    |0.051 +/- 0.0 |0.327 +/- 0.0|