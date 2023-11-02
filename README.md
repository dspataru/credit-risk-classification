# Credit Risk Classification
Train and evaluating a model to classify loan risk

![header_image](https://github.com/dspataru/credit-risk-classification/assets/61765352/2697c809-7838-4fab-8129-b2a630a3c0cb)

## Table of Contents
* [Background](https://github.com/dspataru/credit-risk-classification/blob/main/README.md#background)
* [Overview of the Analysis](https://github.com/dspataru/credit-risk-classification/blob/main/README.md#overview-of-the-analysis)
* [Results](https://github.com/dspataru/credit-risk-classification/blob/main/README.md#results)
* [Summary](https://github.com/dspataru/credit-risk-classification/blob/main/README.md#summary)

## Background

Loan risk, also known as credit risk, refers to the potential for a borrower to fail to meet their obligations under a loan agreement. When a lender provides funds to a borrower, there is always a degree of uncertainty regarding whether the borrower will repay the loan as agreed. Loan risk analysis is the process of evaluating this risk to make informed lending decisions. The purpose of performing a loan risk analysis is to assess the likelihood that a borrower will default on their loan and the potential impact of that default on the lender. By understanding and managing loan risk, lenders can maintain a healthy and profitable loan portfolio while borrowers can access the financing they need.

Machine learning (ML) models can be powerful tools for classifying high-risk and low-risk loans by analyzing various data points and patterns. ML models can process and analysis large amounts of data and find complex patterns and relationships that may not be obvious through traditional or manual analysis. The models are flexible in the sense that they can be customized and update to meet business needs, and can continuously evolve over time from learning the new data. However, it's important to note that the performance and fairness of these models are contingent on the quality and representativeness of the training data.

#### Key Words
Machine learning, classification models, logistic regression, credit risk classification, accuracy score, balanced accuracy score, confusion matrix, classification report, model performance, training set, testing set, train test split, sklearn, pandas, numpy, pathlib

## Overview of the Analysis

In this project, we use various techniques to train and evaluate a ML model based on loan risk. The project uses a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. The data is stored in a csv file and contains the following features:
1. Loan size: total borrower loan amount in dollars.
2. Interest rate: interest rate on the loan in %.
3. Borrower income: the yearly income of the borrower in dollars.
4. Debt to income ratio: the total debt a borrower has divided by their yearly income.
5. Number of accounts: 
6. Derogatory marks: this refers to a negative item that can be found on credit reports sich as late payments or foreclosure. Having a higher number of derogatory marks can hurt your credit score and can affect your loan qualification process. This column in the csv file is stored as an integer.
7. Total debt: total amount of debt a borrower has in dollars.
8. Loan status: 0 or 1

The loan status is the target variable the model is trying to classify. A value of 0 in the “loan_status” column means that the loan is healthy. A value of 1 means that the loan has a high risk of defaulting. A loan default occurs when a borrower fails to meet their financial obligations as outlined in the loan agreement. In other words, it means that the borrower is not making the required payments on the loan according to the terms and conditions specified in the loan contract. The model in this project aims to classify the high-risk loans.

The csv file converted into a pandas dataframe using pd.read_csv. The data was checked for NaN values and no missing entries were found. The data was split into a features dataframe and a labels array, where the features are items 1-7 as detailed above, and loan status contains the labels for the features. The number of healthy loans in the dataset is 75,036 and the number of high risk loans in the dataset is 2,500.

Before creating the ML model, the data is split into training and testing sets using train_test_split from the sklearn library. A random state of 1 was assigned to the function. The size of training set is (58152, 7) which contains 75% of the data from the dataset, and the size of testing set is (19384, 7) which contains the remaining 25% of the data. 

A logistic regression classifier model was used from the sklearn.linear_model library with the following parameters: 
``` python
classifier = LogisticRegression(solver = 'lbfgs',
                                max_iter = 200,
                                random_state = 1)
classifier.fit(X_train, y_train)
```
and the model was fit to the training data. Logistic regression is a fundamental and widely used machine learning technique due to its simplicity, interpretability, and effectiveness in many real-world binary classification tasks. It is a type of regression analysis, but unlike linear regression, which is used for predicting continuous outcomes, logistic regression is specifically designed for predicting binary outcomes, such as "yes" or "no," "spam" or "not spam," "fraudulent" or "non-fraudulent," etc. In our case, we want to predict "high-risk" or "healthy" loans. Below is a visual representation of the logistic regression model and how it uses the sigmoid function to transform the output of a linear equation into a value between 0 and 1 (resource: [logistic regression](https://datahacker.rs/004-machine-learning-logistic-regression-model/)).

![logistic_regression_image](https://github.com/dspataru/credit-risk-classification/assets/61765352/9b73af8f-f71c-476d-8f41-7f076a792a2c)

Let's go back to the dataset. We saw that the number of healthy loans in the dataset is 75,036 and the number of high risk loans in the dataset is 2,500. Based on this information, there is a high number of healthy loans for the model to learn from, but a relatively small number of high risk loans. The data used to train the model is imbalanced. Having an imbalanced dataset can cause a number of issues, including:
1. The model to be biased towards the majority class, which is healthy loans in this case.
2. It can cause poor generaliziation of new data where it could potentially overfit the majority class, resulting in poor performance of the minority class.
3. Can cause inaccurate evaluation.
4. Potential increase of false negatives; instances where the model incorrectly predicts the negative class when it should have predicted the positive class. This may have severe consquences in particular applications and industries.

The are various strategies that can be employed to address the issues that can arise from imbalanced datasets, including evaluation metrics, collecting more data, and resampling techniques. One of the ways to assess a model's performance is by checking the accuracy. Relying solely on accuracy as a performance metric for machine learning models can be problematic and misleading for several reasons, especially when dealing with imbalanced datasets or specific use cases. High accuracy can give a false impression of a model's performance, even if it performs poorly on the class of interest. It may not reflect the model's ability to make accurate predictions for the specific task it was designed for. Instead, there are other evaluation metrics that can be used such as precision, recall, F1-score, ROC AUC, or PR AUC that account for class imbalance, rather than relying solely on accuracy.

In this project, we use the `RandomOverSampler` module from the imbalanced-learn library to resample the data and compare the accuracy and evaluation metrics between the logistic regression model performance on imbalanced vs balanced data.

To evaluate the performance of the models, the following was calculated:
* The balanced accuracy score of the model.
* Confusion matrix.
* Classification report.

The `balanced_accuracy_score` function from the `sklearn.metrics` library is used to compute the balanced accuracy of a classification model. It is a performance metric designed to address the issues associated with imbalanced datasets. Balanced accuracy takes into account the class imbalance by providing an accuracy measure that gives equal weight to each class, making it particularly useful when evaluating models on imbalanced datasets. Below is the process of how computing the `balanced_accuracy_score` works:
1. It computes the true positives (TP), which is class 1, and true negatives (TN), which is class 0, separately.
2. It then calculates the sensitivity for each class, also known as recall, which measures the model's ability to correctly classify positive instances.
3. Next, the class weights are computed: balanced accuracy assigns weights to each class based on their proportion in the dataset. The weights are equal to the number of true negatives for each class divided by the total number of true negatives.
4. Lastly, the balanced accuracy is computed as the arithmetic mean of the sensitivities (recall) for each class, weighted by the class weights.

The result is a score between 0 and 1 where 1 is perfect classification and 0 is "random" classification.

A confusion matrix is a tabular representation of the performance of a classification model, showing the number of correct and incorrect predictions made by the model on a set of data. It is a useful tool for assessing the performance of a classification algorithm and understanding the types of errors it makes. Below is a visual representation of the confusion matrix (resource: [confusion matrix](https://www.google.com/url?sa=i&url=https%3A%2F%2Fmedium.com%2Fanalytics-vidhya%2Fwhat-is-a-confusion-matrix-d1c0f8feda5&psig=AOvVaw3Qh913ADCKpFHEscuXdvjP&ust=1698978354179000&source=images&cd=vfe&opi=89978449&ved=0CBQQjhxqFwoTCPDZ_N-hpIIDFQAAAAAdAAAAABAE)). 

![image](https://github.com/dspataru/credit-risk-classification/assets/61765352/ea5ef816-984d-4d15-af31-b72f11b02af6)

The last method that is used in this project to evaluate the model's performance is the `classification_report`. The classification report will provide a summary of precision, recall, F1-score, and support for each class, along with average scores. Precision is a measure of what proportion of positive identifications was correct. A model that has no false positives has a precision of 1.0. Recall attempts to answer the question of what proportion of actual positives was identified correctly. A model that produces no false negatives has a recall of 1.0. The F1-score is the harmonic mean of precision and recall. It provides a balance between these two metrics, helping to gauge the trade-off between false positives and false negatives.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: Unbalanced Dataset
A prediction was made with the logistic regression model using the testing data. Below is a snapshot of the first 15 predictions the model made using the testing data.

![model_predicitions_OG_data](https://github.com/dspataru/credit-risk-classification/assets/61765352/2e796d9b-ad93-432e-bc36-96be74baacef)

  * Balanced accuracy score: 0.9520479254722232
  * Confusion matrix: ```[[18663   102] [   56   563]] ```
  * Classification report:
    
![imbalanced_classification_report](https://github.com/dspataru/credit-risk-classification/assets/61765352/b72c29e5-2ea8-4701-9d2c-c0eceed3b813)

The balanced accuracy score was calculated for the model to be ~95%. The balanced accuracy in binary classification problems is used to deal with imbalanced dataset. In the case of our model, the dataset has a much higher number of healthy loan status data points than high-risk loan data points, thus we use the balanced accuracy score from the sklearn library vs the accuracy score. 

Based on the classification report, the model has a high precision for predicting the healthy loans. In the case of the regression model, the precision is 1.0, which means that when it predicts the loan_status to be healthy, it is correct 100% of the time. On the same note, when our model predicts the loan status to be high-risk, it is correct 85% of the time. 

In terms of recall, both healthy and high-risk recall scores are high. The model is able to correctly identify 91% of high-risk loans, and 99% of healthy loans.
    
* Machine Learning Model 2: Balanced Dataset
To balance the dataset, `RandomOverSampler` (ROS) was imported from the `imblearn.over_sampling` library. The original training data was fit to the random oversampler model to generate a new dataset. After applying this function, the data was split into an even 56,271 for class 0 and 56,271 for class 1. A logistic regression model was fit to the new training data and the original testing data was used to make predictions as seen below.

![ROS_predictions](https://github.com/dspataru/credit-risk-classification/assets/61765352/58a38648-a4dc-499b-b9cf-3ef3d94704c1)

Similar to the previous model, the ROS model's performance was evaulated using the same metrics.
  * Accuracy score: 0.9938093272802311
  * Confusion matrix: ``` [[18649   116] [    4   615]] ```
  * Classification report:
    
![ROS_classification_report](https://github.com/dspataru/credit-risk-classification/assets/61765352/6fa3ab86-70a8-4832-a5f6-26a7c1a8efd7)


## Summary

![classification_report_comparison](https://github.com/dspataru/credit-risk-classification/assets/61765352/c7d119b5-6517-4639-a5c8-689867580f30)

When using the oversampled data, the logistic regression model has a higher recall for the high-risk loans, resulting in a better f1-score. The precision for the high-risk classification has decreased slightly. The model has an almost 100% precision and recall for both the healthy and high-risk loan data points. The phenomenon that is seen where recall improves and precision decreases is due to the nature of oversampling and the trade-offs involved in managing class imbalance. 

Why does this happen? The goal of over-sampling is to ensure that the model has enough data for the minority class to learn its underlying patterns and improve its ability to correctly classify those instances. By oversampling the minority class, you increase the number of TP predictions because the model now has more examples of the minority class to learn from. As a result, recall tends to improve. Recall is calculated as TP / (TP + FN). Precision may decrease because the model now has more data points to make false positive predictions. When the minority class is oversampled, there is a higher chance the model incorrectly classifies some minority class examples, leading to more FP. Precision is calculated as TP / (TP + FP), so an increase in FP while TP remains relatively constant or grows can lower the precision score.

This trade-off between precision and recall is a common phenomenon in imbalanced classification. By oversampling the minority class, you increase the model's ability to detect the minority class (improved recall), but this comes at the cost of potentially misclassifying more majority class instances as the minority class (reduced precision) as we see in our comparison.

In the case of the application of the model, it is objectively more important to have a higher recall, as you want to reduce the risk of failing to detect a TP (class 1). For companies that are providing loans to borrowers, it would be important to maximize recall because missing a positive case could have serious consequences. For this reason, Machine Learning Model 2: Balanced Dataset is the preferred classification model for the application.  
