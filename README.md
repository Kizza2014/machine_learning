# machine_learning

## Final term project:

#### 1. KNN
- Baseline:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0.0   | 0.91      | 0.99   | 0.95    | 45957   |
    | 1.0   | 0.44      | 0.05   | 0.09    | 4779    |
    | Accuracy |         |        | 0.90    | 50736   |
    | Macro avg | 0.68   | 0.52   | 0.52    | 50736   |
    | Weighted avg | 0.87 | 0.90   | 0.87    | 50736   |

- Random oversampling:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0.0   | 0.97      | 0.73   | 0.83    | 45957   |
    | 1.0   | 0.23      | 0.79   | 0.36    | 4779    |
    | Accuracy |         |        | 0.73    | 50736   |
    | Macro avg | 0.60   | 0.76   | 0.59    | 50736   |
    | Weighted avg | 0.90 | 0.73   | 0.79    | 50736   |

- SMOTE:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0.0   | 0.97      | 0.74   | 0.84    | 45957   |
    | 1.0   | 0.23      | 0.77   | 0.36    | 4779    |
    | Accuracy |         |        | 0.74    | 50736   |
    | Macro avg | 0.60   | 0.75   | 0.60    | 50736   |
    | Weighted avg | 0.90 | 0.74   | 0.79    | 50736   |

- ADASYN:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0.0   | 0.97      | 0.73   | 0.83    | 45957   |
    | 1.0   | 0.23      | 0.78   | 0.35    | 4779    |
    | Accuracy |         |        | 0.74    | 50736   |
    | Macro avg | 0.60   | 0.75   | 0.59    | 50736   |
    | Weighted avg | 0.90 | 0.73   | 0.79    | 50736   |

#### 2. Logistic regression
- Baseline:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0.0   | 0.97      | 0.75   | 0.85    | 45957   |
    | 1.0   | 0.25      | 0.80   | 0.38    | 4779    |
    | Accuracy |         |        | 0.75    | 50736   |
    | Macro avg | 0.61   | 0.77   | 0.61    | 50736   |
    | Weighted avg | 0.90 | 0.75   | 0.80    | 50736   |

- Random oversampling:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0.0   | 0.97      | 0.75   | 0.85    | 45957   |
    | 1.0   | 0.25      | 0.80   | 0.38    | 4779    |
    | Accuracy |         |        | 0.73    | 50736   |
    | Macro avg | 0.61   | 0.77   | 0.61    | 50736   |
    | Weighted avg | 0.90 | 0.75   | 0.80    | 50736   |

- SMOTE:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0.0   | 0.97      | 0.75   | 0.85    | 45957   |
    | 1.0   | 0.23      | 0.78   | 0.38    | 4779    |
    | Accuracy |         |        | 0.76    | 50736   |
    | Macro avg | 0.61   | 0.77   | 0.61    | 50736   |
    | Weighted avg | 0.90 | 0.76   | 0.80    | 50736   |

- ADASYN:
    | Class | Precision | Recall | F1-score | Support |
    |-------|-----------|--------|----------|---------|
    | 0.0   | 0.97      | 0.74   | 0.84    | 45957   |
    | 1.0   | 0.24      | 0.80   | 0.37    | 4779    |
    | Accuracy |         |        | 0.75    | 50736   |
    | Macro avg | 0.61   | 0.77   | 0.61    | 50736   |
    | Weighted avg | 0.90 | 0.75   | 0.80    | 50736   |
