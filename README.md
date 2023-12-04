# CS444-VQA
## Baseline
### Simple VAQ dataset
The dataset is in [/dataSimpleVQA](/dataSimpleVQA/)

This dataset has 13 answers.

There are 3 types of questions: Yes/No, What Shape, and What Color.

| Model         | Accuracy |
| ------------- | ------------- |
| random        |  Best 36.89% when select "no" only |
|               |  Accuracy decrease by increasing number of random answers to choose|
|               |  8.9% when select randomly from all 13 answers|
| prior yes     |  36.63%|
| prior Q-type prior  | 43.26%  |
| KNN  |   |

KNN results when K=4
Test set: Accuracy: 1490/9468 (15.74%)

### VAQ dataset