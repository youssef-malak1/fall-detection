# Feature-group ablation

Same 5-fold event-level CV for every row; only the input features change.

| features | model | precision | recall | F1 |
|---|---|---|---|---|
| motion | logreg | 0.900 | 0.867 | 0.858 ± 0.109 |
| motion | rf | 0.833 | 0.800 | 0.791 ± 0.104 |
| pose | logreg | 0.896 | 0.867 | 0.861 ± 0.143 |
| pose | rf | 0.892 | 0.800 | 0.833 ± 0.073 |
| motion+pose | logreg | 0.950 | 0.933 | 0.935 ± 0.056 |
| motion+pose | rf | 0.888 | 0.833 | 0.838 ± 0.092 |
