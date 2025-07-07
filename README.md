# DECISION-TREE
COMPANY: CODTECH IT SOLUTIONS

NAME: BHARAT BHANDARI

INTERN ID: CT04DF123

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH


---


##  TASK 1 – DECISION TREE CLASSIFIER

This repository contains my submission for **Task 1** of the **CodTech Machine Learning Internship**.

The goal of this task was to:
> Build a **Decision Tree Classifier** using Scikit-Learn, visualize it completely using `plot_tree()`, and evaluate its performance (on full data).

---

##  OBJECTIVE

-  Load and explore the Iris dataset  
-  Train a decision tree model on the **entire dataset**
-  Visualize the decision-making structure of the model
-   Generate performance metrics (accuracy & classification report)
-   ep the notebook beginner-friendly and explain each step  

---

## 📊 DATASET OVERVIEW

**Dataset Used**: Iris Dataset (from `sklearn.datasets`)

Samples : 150

Features : 4

Target Classes: 3


***
###  Feature Columns:
- Sepal Length (cm)  
- Sepal Width (cm)  
- Petal Length (cm)  
- Petal Width (cm)

###  Target Labels:
- Iris Setosa  
- Iris Versicolor  
- Iris Virginica  

The dataset is **balanced and clean**, perfect for learning tree-based models.

---

## ⚙ LIBRARIES USED

```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
```
***

##  STEPS PERFORMED
### 1. Load the Dataset
```python
iris_data = load_iris()
features = iris_data.data
labels = iris_data.target
```
### 2. Train the Decision Tree Classifier
```python
tree_model = DecisionTreeClassifier(random_state=0)
tree_model.fit(features, labels)

```
We used the entire dataset for training (no split) to show a full tree.

### 3. Evaluate the Model (on training data)
```python
predictions = tree_model.predict(features)

accuracy = accuracy_score(labels, predictions)
print("Accuracy:", accuracy)

print(classification_report(labels, predictions, target_names=iris_data.target_names))
```
### 4. Visualize the Tree
```python
plt.figure(figsize=(14, 10))
plot_tree(tree_model,
          feature_names=iris_data.feature_names,
          class_names=iris_data.target_names,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Decision Tree (Trained on Full Dataset)")
plt.show()

```
***
The visualization shows:

- Feature & threshold used at each split

- Gini score for node purity

- Number of samples per node

- Predicted class at each leaf
*** 
## TREE OUTPUT
![Trained Decision Tree Plot showing splits based on petal width and length](decision-tree.png)
***
This plot helps us see the logic of how the model splits and classifies each flower type based on petal/sepal measurements.
***

# CONCLUSION
- Successfully built a decision tree classifier using the Iris dataset

- Achieved 100% accuracy on training data

- Visualized model decision-making process clearly

- Gained a solid understanding of tree-based classification as a beginner








