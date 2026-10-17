# Comparison — elliptic

Research benchmark: **True**.

| Model | AUROC | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | --- | --- |
| logistic_regression | 0.5894 | 0.1070 | 0.9751 | 0.1928 |  |
| svm | 0.6139 | 0.1044 | 0.9052 | 0.1873 |  |
| random_forest | 0.7631 | 0.1978 | 0.7556 | 0.3135 |  |
| static_gcn | 0.8628 | 0.2722 | 0.6658 | 0.3864 |  |
| static_gat | 0.7772 | 0.1649 | 0.8279 | 0.2751 |  |
| gnn_only | 0.8246 | 0.2525 | 0.5661 | 0.3492 |  |
| lstm_only | n/a | n/a | n/a | n/a | This dataset is a node graph, not a per-account transaction sequence, so the LSTM autoencoder is not applicable. |
| fused_lstm_gnn | n/a | n/a | n/a | n/a | Fusion needs an LSTM sequence score. It is reported on the IEEE-CIS transaction stream. |
