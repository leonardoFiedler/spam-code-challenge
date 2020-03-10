## Second Stage

### Description

The purpose of this exercise is to classify data automatically into spam and common messages. 
To solve this, it was used 2 algorithms: 

1. [Random Forest][random_forest_link]
2. [Decision Tree][decision_tree_link].

[random_forest_link]: https://towardsdatascience.com/understanding-random-forest-58381e0602d2
[decision_tree_link]: https://towardsdatascience.com/decision-tree-in-machine-learning-e380942a4c96

### Command

To execute this file, you can run with the following command: python second_stage.py.

### Parameters

|Parameter      |Meaning      |Accepted Values      |Required |
|---------------|-------------|---------------------|---------|
|option (-o)    |Number of algorithm listed above or 3 for both|[1-2]|False
|all (-a)       |Execute all algorithms|-|False

#### Attention

At least one parameter should be picked!

### Output

The output of the program are the metrics:

• Mean Absolute Error

• Accuracy

• Confusion Matrix

• Precision

• Recal

• F1
