Implementing GZip Lasso Regression (with kNN=0) shows worse performance then using normal base regression with kNN. 
The alpha used was 0.03. I used GridSearch to find the best alpha. However, I only did it
for freesolv in this iteration.

# Results Gzip-based Molecular Lasso-Regression
|Data Set|Split |AUROC/RMSE (Valid)|F1/MAE (Valid) |AUROC/RMSE (Test)| F1/MAE (Test) |
|--------|------|------------------|---------------|-----------------|---------------|
|freesolv|random|0.751 +/- 0.174   |0.341 +/- 0.174|0.653 +/- 0.174  |0.367 +/- 0.174|
|delaney |random|1.562 +/- 0.051   |1.236 +/- 0.051|1.693 +/- 0.051  |1.337 +/- 0.051|
|lipo    |random|1.105 +/- 0.035   |0.925 +/- 0.035|1.134 +/- 0.035  |0.954 +/- 0.035|

Also, tried adding in categorical information from what unique smiles x1 and x2 came from to form the compressed
sequence. It definitely hurt the model. Still wanted to try it out.
freesolv alpha = 0.02
delaney alpha = 0.0 (Lasso doesn't converge well with an alpha of 0.0)
lip alpha = 0.01
# Results Gzip-based Molecular Lasso-Regression w/ OneHot
|Data Set|Split |AUROC/RMSE (Valid)|F1/MAE (Valid) |AUROC/RMSE (Test)| F1/MAE (Test) |
|--------|------|------------------|---------------|-----------------|---------------|
|freesolv|random|0.626 +/- 0.076   |0.331 +/- 0.076|0.674 +/- 0.076  |0.328 +/- 0.076|
|delaney |random|1.705 +/- 0.094   |1.33 +/- 0.094 |1.706 +/- 0.094  |1.334 +/- 0.094|
|lipo    |random|1.132 +/- 0.022   |0.949 +/- 0.022|1.124 +/- 0.022  |0.93  +/- 0.022|

