# Comparison — synthetic-fixture

Research benchmark: **False**.

| Model | AUROC | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.9831 | 0.8571 | 0.9231 | 0.8889 |  |
| svm | 0.9831 | 1.0000 | 0.6923 | 0.8182 |  |
| random_forest | 0.9250 | 0.6000 | 0.6923 | 0.6429 |  |
| static_gcn | 0.5385 | 0.2292 | 0.8462 | 0.3607 |  |
| static_gat | 0.4259 | 0.2000 | 0.7692 | 0.3175 |  |
| gnn_only | 0.4897 | 0.2444 | 0.8462 | 0.3793 |  |
| lstm_only | 0.6745 | 0.2407 | 1.0000 | 0.3881 |  |
| fused_lstm_gnn | 0.4897 | 0.2444 | 0.8462 | 0.3793 |  |
