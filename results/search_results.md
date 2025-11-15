# Results Gzip-based Molecular Classification
|Data Set|     Task     | Split  |AUROC/RMSE (Valid)|F1/MAE (Valid)|-/R (Valid)|AUROC/RMSE (Test)|F1/MAE (Test)| -/R (Test)  |compressor|transform|
|--------|--------------|--------|------------------|--------------|-----------|-----------------|-------------|-------------|----------|---------|
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.648 +/- 0.0    |0.639 +/- 0.0|0.0 +/- 0.0  |Gzip      |None     |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.594 +/- 0.0    |0.17 +/- 0.0 |0.0 +/- 0.0  |Gzip      |None     |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.579 +/- 0.0    |0.66 +/- 0.0 |0.0 +/- 0.0  |Gzip      |None     |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.914 +/- 0.0    |0.932 +/- 0.0|0.0 +/- 0.0  |Gzip      |None     |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.035 +/- 0.0    |0.829 +/- 0.0|0.422 +/- 0.0|Gzip      |None     |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.325 +/- 0.0    |1.006 +/- 0.0|0.722 +/- 0.0|Gzip      |None     |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.754 +/- 0.0    |2.864 +/- 0.0|0.442 +/- 0.0|Gzip      |None     |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.688 +/- 0.0    |0.886 +/- 0.0|0.0 +/- 0.0  |Gzip      |None     |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.045 +/- 0.0    |0.028 +/- 0.0|0.904 +/- 0.0|Gzip      |None     |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.691 +/- 0.0    |0.68 +/- 0.0 |0.0 +/- 0.0  |LZ4       |None     |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.587 +/- 0.0    |0.179 +/- 0.0|0.0 +/- 0.0  |LZ4       |None     |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.584 +/- 0.0    |0.651 +/- 0.0|0.0 +/- 0.0  |LZ4       |None     |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.87 +/- 0.0     |0.942 +/- 0.0|0.0 +/- 0.0  |LZ4       |None     |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.979 +/- 0.0    |0.788 +/- 0.0|0.491 +/- 0.0|LZ4       |None     |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.46 +/- 0.0     |1.016 +/- 0.0|0.654 +/- 0.0|LZ4       |None     |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|5.586 +/- 0.0    |4.634 +/- 0.0|0.188 +/- 0.0|LZ4       |None     |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.687 +/- 0.0    |0.856 +/- 0.0|0.0 +/- 0.0  |LZ4       |None     |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.047 +/- 0.0    |0.03 +/- 0.0 |0.898 +/- 0.0|LZ4       |None     |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.641 +/- 0.0    |0.629 +/- 0.0|0.0 +/- 0.0  |Snappy    |None     |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.575 +/- 0.0    |0.164 +/- 0.0|0.0 +/- 0.0  |Snappy    |None     |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.594 +/- 0.0    |0.649 +/- 0.0|0.0 +/- 0.0  |Snappy    |None     |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.882 +/- 0.0    |0.963 +/- 0.0|0.0 +/- 0.0  |Snappy    |None     |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.072 +/- 0.0    |0.855 +/- 0.0|0.381 +/- 0.0|Snappy    |None     |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.736 +/- 0.0    |1.318 +/- 0.0|0.599 +/- 0.0|Snappy    |None     |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|5.343 +/- 0.0    |4.46 +/- 0.0 |0.306 +/- 0.0|Snappy    |None     |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.667 +/- 0.0    |0.838 +/- 0.0|0.0 +/- 0.0  |Snappy    |None     |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.05 +/- 0.0     |0.031 +/- 0.0|0.885 +/- 0.0|Snappy    |None     |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.638 +/- 0.0    |0.629 +/- 0.0|0.0 +/- 0.0  |Gzip      |LZ4      |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.587 +/- 0.0    |0.176 +/- 0.0|0.0 +/- 0.0  |Gzip      |LZ4      |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.575 +/- 0.0    |0.637 +/- 0.0|0.0 +/- 0.0  |Gzip      |LZ4      |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.92 +/- 0.0     |0.942 +/- 0.0|0.0 +/- 0.0  |Gzip      |LZ4      |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.031 +/- 0.0    |0.819 +/- 0.0|0.439 +/- 0.0|Gzip      |LZ4      |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.203 +/- 0.0    |0.947 +/- 0.0|0.77 +/- 0.0 |Gzip      |LZ4      |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.734 +/- 0.0    |2.768 +/- 0.0|0.557 +/- 0.0|Gzip      |LZ4      |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.699 +/- 0.0    |0.893 +/- 0.0|0.0 +/- 0.0  |Gzip      |LZ4      |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.044 +/- 0.0    |0.027 +/- 0.0|0.91 +/- 0.0 |Gzip      |LZ4      |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.646 +/- 0.0    |0.634 +/- 0.0|0.0 +/- 0.0  |LZ4       |LZ4      |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.589 +/- 0.0    |0.176 +/- 0.0|0.0 +/- 0.0  |LZ4       |LZ4      |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.586 +/- 0.0    |0.651 +/- 0.0|0.0 +/- 0.0  |LZ4       |LZ4      |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.88 +/- 0.0     |0.959 +/- 0.0|0.0 +/- 0.0  |LZ4       |LZ4      |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.052 +/- 0.0    |0.836 +/- 0.0|0.421 +/- 0.0|LZ4       |LZ4      |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.781 +/- 0.0    |1.327 +/- 0.0|0.496 +/- 0.0|LZ4       |LZ4      |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|5.969 +/- 0.0    |4.931 +/- 0.0|0.191 +/- 0.0|LZ4       |LZ4      |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.689 +/- 0.0    |0.851 +/- 0.0|0.0 +/- 0.0  |LZ4       |LZ4      |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.046 +/- 0.0    |0.029 +/- 0.0|0.9 +/- 0.0  |LZ4       |LZ4      |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.607 +/- 0.0    |0.598 +/- 0.0|0.0 +/- 0.0  |Snappy    |LZ4      |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.567 +/- 0.0    |0.142 +/- 0.0|0.0 +/- 0.0  |Snappy    |LZ4      |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.594 +/- 0.0    |0.665 +/- 0.0|0.0 +/- 0.0  |Snappy    |LZ4      |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.901 +/- 0.0    |0.907 +/- 0.0|0.0 +/- 0.0  |Snappy    |LZ4      |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.032 +/- 0.0    |0.819 +/- 0.0|0.401 +/- 0.0|Snappy    |LZ4      |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.849 +/- 0.0    |1.453 +/- 0.0|0.576 +/- 0.0|Snappy    |LZ4      |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|5.573 +/- 0.0    |4.629 +/- 0.0|0.035 +/- 0.0|Snappy    |LZ4      |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.682 +/- 0.0    |0.881 +/- 0.0|0.0 +/- 0.0  |Snappy    |LZ4      |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.048 +/- 0.0    |0.03 +/- 0.0 |0.892 +/- 0.0|Snappy    |LZ4      |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.642 +/- 0.0    |0.634 +/- 0.0|0.0 +/- 0.0  |Gzip      |Snappy   |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.577 +/- 0.0    |0.155 +/- 0.0|0.0 +/- 0.0  |Gzip      |Snappy   |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.584 +/- 0.0    |0.652 +/- 0.0|0.0 +/- 0.0  |Gzip      |Snappy   |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.693 +/- 0.0    |0.796 +/- 0.0|0.0 +/- 0.0  |Gzip      |Snappy   |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.027 +/- 0.0    |0.826 +/- 0.0|0.439 +/- 0.0|Gzip      |Snappy   |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.265 +/- 0.0    |0.951 +/- 0.0|0.739 +/- 0.0|Gzip      |Snappy   |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.139 +/- 0.0    |2.389 +/- 0.0|0.638 +/- 0.0|Gzip      |Snappy   |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.705 +/- 0.0    |0.883 +/- 0.0|0.0 +/- 0.0  |Gzip      |Snappy   |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.046 +/- 0.0    |0.029 +/- 0.0|0.899 +/- 0.0|Gzip      |Snappy   |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.584 +/- 0.0    |0.577 +/- 0.0|0.0 +/- 0.0  |LZ4       |Snappy   |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.575 +/- 0.0    |0.158 +/- 0.0|0.0 +/- 0.0  |LZ4       |Snappy   |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.553 +/- 0.0    |0.638 +/- 0.0|0.0 +/- 0.0  |LZ4       |Snappy   |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.672 +/- 0.0    |0.846 +/- 0.0|0.0 +/- 0.0  |LZ4       |Snappy   |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.061 +/- 0.0    |0.865 +/- 0.0|0.413 +/- 0.0|LZ4       |Snappy   |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.429 +/- 0.0    |1.109 +/- 0.0|0.669 +/- 0.0|LZ4       |Snappy   |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.625 +/- 0.0    |2.643 +/- 0.0|0.343 +/- 0.0|LZ4       |Snappy   |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.708 +/- 0.0    |0.873 +/- 0.0|0.0 +/- 0.0  |LZ4       |Snappy   |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.047 +/- 0.0    |0.03 +/- 0.0 |0.894 +/- 0.0|LZ4       |Snappy   |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.57 +/- 0.0     |0.567 +/- 0.0|0.0 +/- 0.0  |Snappy    |Snappy   |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.558 +/- 0.0    |0.151 +/- 0.0|0.0 +/- 0.0  |Snappy    |Snappy   |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.585 +/- 0.0    |0.662 +/- 0.0|0.0 +/- 0.0  |Snappy    |Snappy   |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.637 +/- 0.0    |0.825 +/- 0.0|0.0 +/- 0.0  |Snappy    |Snappy   |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.092 +/- 0.0    |0.853 +/- 0.0|0.382 +/- 0.0|Snappy    |Snappy   |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.514 +/- 0.0    |1.157 +/- 0.0|0.611 +/- 0.0|Snappy    |Snappy   |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.961 +/- 0.0    |3.115 +/- 0.0|0.271 +/- 0.0|Snappy    |Snappy   |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.67 +/- 0.0     |0.895 +/- 0.0|0.0 +/- 0.0  |Snappy    |Snappy   |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.047 +/- 0.0    |0.03 +/- 0.0 |0.895 +/- 0.0|Snappy    |Snappy   |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.634 +/- 0.0    |0.624 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 1    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.568 +/- 0.0    |0.172 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 1    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.599 +/- 0.0    |0.656 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 1    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.865 +/- 0.0    |0.932 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 1    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.977 +/- 0.0    |0.781 +/- 0.0|0.496 +/- 0.0|Gzip      |Aug 1    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.101 +/- 0.0    |0.823 +/- 0.0|0.812 +/- 0.0|Gzip      |Aug 1    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.458 +/- 0.0    |2.64 +/- 0.0 |0.495 +/- 0.0|Gzip      |Aug 1    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.7 +/- 0.0      |0.866 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 1    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.043 +/- 0.0    |0.027 +/- 0.0|0.908 +/- 0.0|Gzip      |Aug 1    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.659 +/- 0.0    |0.649 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 1    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.589 +/- 0.0    |0.154 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 1    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.617 +/- 0.0    |0.684 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 1    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.852 +/- 0.0    |0.908 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 1    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.048 +/- 0.0    |0.845 +/- 0.0|0.396 +/- 0.0|LZ4       |Aug 1    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.151 +/- 0.0    |0.923 +/- 0.0|0.799 +/- 0.0|LZ4       |Aug 1    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.849 +/- 0.0    |2.932 +/- 0.0|0.463 +/- 0.0|LZ4       |Aug 1    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.714 +/- 0.0    |0.885 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 1    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.047 +/- 0.0    |0.03 +/- 0.0 |0.901 +/- 0.0|LZ4       |Aug 1    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.638 +/- 0.0    |0.629 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 1    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.571 +/- 0.0    |0.167 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 1    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.579 +/- 0.0    |0.644 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 1    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.785 +/- 0.0    |0.918 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 1    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.029 +/- 0.0    |0.831 +/- 0.0|0.406 +/- 0.0|Snappy    |Aug 1    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.074 +/- 0.0    |0.888 +/- 0.0|0.838 +/- 0.0|Snappy    |Aug 1    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|4.239 +/- 0.0    |3.217 +/- 0.0|0.332 +/- 0.0|Snappy    |Aug 1    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.713 +/- 0.0    |0.891 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 1    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.047 +/- 0.0    |0.03 +/- 0.0 |0.901 +/- 0.0|Snappy    |Aug 1    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.664 +/- 0.0    |0.655 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 3    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.593 +/- 0.0    |0.182 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 3    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.612 +/- 0.0    |0.669 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 3    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.824 +/- 0.0    |0.905 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 3    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.99 +/- 0.0     |0.794 +/- 0.0|0.481 +/- 0.0|Gzip      |Aug 3    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.092 +/- 0.0    |0.81 +/- 0.0 |0.821 +/- 0.0|Gzip      |Aug 3    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.157 +/- 0.0    |2.273 +/- 0.0|0.595 +/- 0.0|Gzip      |Aug 3    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.711 +/- 0.0    |0.888 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 3    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.042 +/- 0.0    |0.026 +/- 0.0|0.916 +/- 0.0|Gzip      |Aug 3    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.656 +/- 0.0    |0.649 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 3    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.57 +/- 0.0     |0.166 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 3    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.592 +/- 0.0    |0.66 +/- 0.0 |0.0 +/- 0.0  |LZ4       |Aug 3    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.784 +/- 0.0    |0.874 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 3    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.983 +/- 0.0    |0.799 +/- 0.0|0.481 +/- 0.0|LZ4       |Aug 3    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.02 +/- 0.0     |0.824 +/- 0.0|0.855 +/- 0.0|LZ4       |Aug 3    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.569 +/- 0.0    |2.61 +/- 0.0 |0.51 +/- 0.0 |LZ4       |Aug 3    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.696 +/- 0.0    |0.873 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 3    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.044 +/- 0.0    |0.028 +/- 0.0|0.908 +/- 0.0|LZ4       |Aug 3    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.65 +/- 0.0     |0.639 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 3    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.584 +/- 0.0    |0.166 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 3    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.585 +/- 0.0    |0.666 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 3    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.678 +/- 0.0    |0.767 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 3    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.01 +/- 0.0     |0.824 +/- 0.0|0.441 +/- 0.0|Snappy    |Aug 3    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.095 +/- 0.0    |0.842 +/- 0.0|0.824 +/- 0.0|Snappy    |Aug 3    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.658 +/- 0.0    |2.774 +/- 0.0|0.491 +/- 0.0|Snappy    |Aug 3    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.686 +/- 0.0    |0.883 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 3    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.044 +/- 0.0    |0.028 +/- 0.0|0.909 +/- 0.0|Snappy    |Aug 3    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.652 +/- 0.0    |0.644 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 5    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.568 +/- 0.0    |0.153 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 5    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.599 +/- 0.0    |0.679 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 5    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.81 +/- 0.0     |0.922 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 5    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.988 +/- 0.0    |0.785 +/- 0.0|0.483 +/- 0.0|Gzip      |Aug 5    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.013 +/- 0.0    |0.776 +/- 0.0|0.846 +/- 0.0|Gzip      |Aug 5    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.221 +/- 0.0    |2.292 +/- 0.0|0.65 +/- 0.0 |Gzip      |Aug 5    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.703 +/- 0.0    |0.885 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 5    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.041 +/- 0.0    |0.025 +/- 0.0|0.918 +/- 0.0|Gzip      |Aug 5    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.639 +/- 0.0    |0.634 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 5    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.57 +/- 0.0     |0.14 +/- 0.0 |0.0 +/- 0.0  |LZ4       |Aug 5    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.598 +/- 0.0    |0.686 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 5    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.749 +/- 0.0    |0.81 +/- 0.0 |0.0 +/- 0.0  |LZ4       |Aug 5    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.01 +/- 0.0     |0.791 +/- 0.0|0.447 +/- 0.0|LZ4       |Aug 5    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.068 +/- 0.0    |0.858 +/- 0.0|0.83 +/- 0.0 |LZ4       |Aug 5    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.983 +/- 0.0    |2.999 +/- 0.0|0.303 +/- 0.0|LZ4       |Aug 5    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.679 +/- 0.0    |0.868 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 5    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.043 +/- 0.0    |0.027 +/- 0.0|0.912 +/- 0.0|LZ4       |Aug 5    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.631 +/- 0.0    |0.624 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 5    |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.602 +/- 0.0    |0.169 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 5    |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.602 +/- 0.0    |0.669 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 5    |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.76 +/- 0.0     |0.785 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 5    |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.95 +/- 0.0     |0.769 +/- 0.0|0.526 +/- 0.0|Snappy    |Aug 5    |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.181 +/- 0.0    |0.902 +/- 0.0|0.787 +/- 0.0|Snappy    |Aug 5    |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|4.22 +/- 0.0     |3.218 +/- 0.0|0.366 +/- 0.0|Snappy    |Aug 5    |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.69 +/- 0.0     |0.882 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 5    |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.044 +/- 0.0    |0.027 +/- 0.0|0.91 +/- 0.0 |Snappy    |Aug 5    |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.682 +/- 0.0    |0.675 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 10   |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.595 +/- 0.0    |0.177 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 10   |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.626 +/- 0.0    |0.692 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 10   |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.802 +/- 0.0    |0.863 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 10   |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.962 +/- 0.0    |0.765 +/- 0.0|0.515 +/- 0.0|Gzip      |Aug 10   |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.937 +/- 0.0    |0.729 +/- 0.0|0.87 +/- 0.0 |Gzip      |Aug 10   |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.158 +/- 0.0    |2.332 +/- 0.0|0.675 +/- 0.0|Gzip      |Aug 10   |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.714 +/- 0.0    |0.886 +/- 0.0|0.0 +/- 0.0  |Gzip      |Aug 10   |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.041 +/- 0.0    |0.025 +/- 0.0|0.918 +/- 0.0|Gzip      |Aug 10   |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.649 +/- 0.0    |0.644 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 10   |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.583 +/- 0.0    |0.16 +/- 0.0 |0.0 +/- 0.0  |LZ4       |Aug 10   |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.606 +/- 0.0    |0.679 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 10   |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.6 +/- 0.0      |0.759 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 10   |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.012 +/- 0.0    |0.815 +/- 0.0|0.456 +/- 0.0|LZ4       |Aug 10   |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.157 +/- 0.0    |0.926 +/- 0.0|0.797 +/- 0.0|LZ4       |Aug 10   |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|4.12 +/- 0.0     |3.139 +/- 0.0|0.381 +/- 0.0|LZ4       |Aug 10   |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.69 +/- 0.0     |0.862 +/- 0.0|0.0 +/- 0.0  |LZ4       |Aug 10   |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.043 +/- 0.0    |0.027 +/- 0.0|0.911 +/- 0.0|LZ4       |Aug 10   |
|bbbp    |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.693 +/- 0.0    |0.691 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 10   |
|tox21   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.583 +/- 0.0    |0.167 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 10   |
|sider   |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.607 +/- 0.0    |0.667 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 10   |
|clintox |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.69 +/- 0.0     |0.836 +/- 0.0|0.0 +/- 0.0  |Snappy    |Aug 10   |
|lipo    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.01 +/- 0.0     |0.813 +/- 0.0|0.438 +/- 0.0|Snappy    |Aug 10   |
|esol    |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|1.253 +/- 0.0    |0.953 +/- 0.0|0.754 +/- 0.0|Snappy    |Aug 10   |
|freesolv|regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|3.948 +/- 0.0    |3.021 +/- 0.0|0.338 +/- 0.0|Snappy    |Aug 10   |
|hiv     |classification|scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.674 +/- 0.0    |0.88 +/- 0.0 |0.0 +/- 0.0  |Snappy    |Aug 10   |
|qm8     |regression    |scaffold|0.0 +/- 0.0       |0.0 +/- 0.0   |0.0 +/- 0.0|0.043 +/- 0.0    |0.027 +/- 0.0|0.913 +/- 0.0|Snappy    |Aug 10   |
