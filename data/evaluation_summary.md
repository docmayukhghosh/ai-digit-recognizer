# Model Evaluation Summary

- **Model Path**: `models\digit_recognizer_model.keras`
- **Test Accuracy**: 98.30%
- **Test Loss**: 0.0535

## Per-Class Metrics

|              |   precision |   recall |   f1-score |   support |
|:-------------|------------:|---------:|-----------:|----------:|
| 0            |      0.9809 |   0.9949 |     0.9878 |   980     |
| 1            |      0.9869 |   0.9938 |     0.9903 |  1135     |
| 2            |      0.9787 |   0.9797 |     0.9792 |  1032     |
| 3            |      0.993  |   0.9782 |     0.9855 |  1010     |
| 4            |      0.976  |   0.9939 |     0.9849 |   982     |
| 5            |      0.98   |   0.9888 |     0.9844 |   892     |
| 6            |      0.9905 |   0.9802 |     0.9853 |   958     |
| 7            |      0.9731 |   0.9854 |     0.9792 |  1028     |
| 8            |      0.9824 |   0.9733 |     0.9778 |   974     |
| 9            |      0.9888 |   0.9613 |     0.9749 |  1009     |
| accuracy     |      0.983  |   0.983  |     0.983  |     0.983 |
| macro avg    |      0.983  |   0.983  |     0.9829 | 10000     |
| weighted avg |      0.9831 |   0.983  |     0.983  | 10000     |

## Insights & Errors
The confusion matrix and misclassified examples plots have been generated and placed in the `/data/` folder.
Common errors in MNIST usually involve digits with overlapping structural styles, such as:
- **4 vs 9**: Incomplete loops at the top of 9s can resemble 4s.
- **3 vs 8**: Missing connections in 8s can resemble 3s.
- **7 vs 1**: Slanted lines can cause confusion between 7s and 1s.
