# Credit Risk Classification
Train and evaluating a model to classify loan risk

![header_image](https://github.com/dspataru/credit-risk-classification/assets/61765352/2697c809-7838-4fab-8129-b2a630a3c0cb)

## Table of Contents

## Background

Loan risk, also known as credit risk, refers to the potential for a borrower to fail to meet their obligations under a loan agreement. When a lender provides funds to a borrower, there is always a degree of uncertainty regarding whether the borrower will repay the loan as agreed. Loan risk analysis is the process of evaluating this risk to make informed lending decisions. The purpose of performing a loan risk analysis is to assess the likelihood that a borrower will default on their loan and the potential impact of that default on the lender. 

The goal of loan risk analysis is to strike a balance between providing access to credit for borrowers while managing the lender's exposure to risk. Lenders aim to minimize the likelihood of loan defaults by making informed decisions and using risk mitigation strategies. By understanding and managing loan risk, lenders can maintain a healthy and profitable loan portfolio while borrowers can access the financing they need.

Machine learning (ML) models can be powerful tools for classifying high-risk and low-risk loans by analyzing various data points and patterns. ML models can process and analysis large amounts of data and find complex patterns and relationships that may not be obvious through traditional or manual analysis. The models are flexible in the sense that they can be customized and update to meet business needs, and can continuously evolve over time from learning the new data.

However, it's important to note that the performance and fairness of these models are contingent on the quality and representativeness of the training data, as well as ethical considerations to avoid bias and discrimination in lending decisions. At the end of the day, ML models are not a replacement for human judgement, but rather, a valuable tool to enhance decision-making processes and pair well with expert knowledge and experience.

#### Key Words
Machine learning, classification models, linear regression, 

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
and the model was fit to the training data. Logistic regression is a fundamental and widely used machine learning technique due to its simplicity, interpretability, and effectiveness in many real-world binary classification tasks. It is a type of regression analysis, but unlike linear regression, which is used for predicting continuous outcomes, logistic regression is specifically designed for predicting binary outcomes, such as "yes" or "no," "spam" or "not spam," "fraudulent" or "non-fraudulent," etc. In our case, we want to predict "high-risk" or "healthy" loans. Below is a visual representation of the logistic regression model and how it uses the sigmoid function to transform the output of a linear equation into a value between 0 and 1 (resource: https://datahacker.rs/004-machine-learning-logistic-regression-model/).

![logistic_regression_image](https://github.com/dspataru/credit-risk-classification/assets/61765352/9b73af8f-f71c-476d-8f41-7f076a792a2c)

Let's go back to the dataset. We saw that the number of healthy loans in the dataset is 75,036 and the number of high risk loans in the dataset is 2,500. Based on this information, there is a high number of healthy loans for the model to learn from, but a relatively small number of high risk loans. The data used to train the model is imbalanced. Having an imbalanced dataset can cause a number of issues, including:
1. The model to be biased towards the majority class, which is healthy loans in this case.
2. It can cause poor generaliziation of new data where it could potentially overfit the majority class, resulting in poor performance of the minority class.
3. Can cause inaccurate evaluation.
4. Potential increase of false negatives; instances where the model incorrectly predicts the negative class when it should have predicted the positive class. This may have severe consquences in particular applications and industries.

The are various strategies that can be employed to address the issues that can arise from imbalanced datasets, including evaluation metrics, collecting more data, and resampling techniques. One of the ways to assess a model's performance is by checking the accuracy. Relying solely on accuracy as a performance metric for machine learning models can be problematic and misleading for several reasons, especially when dealing with imbalanced datasets or specific use cases. High accuracy can give a false impression of a model's performance, even if it performs poorly on the class of interest. It may not reflect the model's ability to make accurate predictions for the specific task it was designed for. Instead, there are other evaluation metrics that can be used such as precision, recall, F1-score, ROC AUC, or PR AUC that account for class imbalance, rather than relying solely on accuracy.

In this project, we explore resampling techniques and compare accuracy and evaluation metrics.

## Results

To evaluate the prediction results


Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores.
A prediction was made with the logistic regression model using the testing data. Below is a snapshot of the first 15 predictions the model made using the testing data.

![model_predicitions_OG_data](https://github.com/dspataru/credit-risk-classification/assets/61765352/2e796d9b-ad93-432e-bc36-96be74baacef)


* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
