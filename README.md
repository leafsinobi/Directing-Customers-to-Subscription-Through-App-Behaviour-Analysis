# Directing-Customers-to-Subscription-Through-App-Behaviour-Analysis


This repository contains Python code for building and evaluating a machine learning 
model to predict user enrollment in an application.The code follows a standard machine learning
pipeline and uses logistic regression with L1 regularization for predictive modeling. 
Below is an overview of the key components of the code:


Overview:


1. Importing Libraries: The code begins by importing essential libraries such as pandas, numpy, seaborn, and scikit-learn. It loads the dataset from 'new_appdata10.csv' for further processing.

2. Data Pre-Processing: The dataset is pre-processed by splitting it into independent features (X) and the target variable (response), which represents user enrollment.
   It further divides the data into training and testing sets (80% training, 20% testing). Additionally, feature scaling is applied using StandardScaler.

3.Model Building: Logistic Regression with L1 regularization is selected as the classification algorithm. The model is fitted to the training data.

4. Model Evaluation: The code evaluates the model's performance using various metrics, including accuracy, precision, recall, and F1-score. It also creates a heatmap of the confusion matrix for visualization.

5. K-Fold Cross-Validation: K-fold cross-validation (k=10) is applied to assess the model's generalization performance and check for overfitting.

6. Model Tuning: Grid search is conducted to optimize hyperparameters for the logistic regression model. Two rounds of grid search are performed to find the best combination of regularization parameters.

7. End of Model: The final results, including actual enrollment status, user identifiers, and predicted enrollment status, are formatted into a DataFrame for further analysis and reporting.


To use this code:


1. Clone this repository to your local machine.
2. Ensure you have Python and the required libraries installed (pandas, numpy, scikit-learn, seaborn, matplotlib).
3. Place your dataset in the same directory as the code and update the dataset filename accordingly.
4. Run the code to train, evaluate, and tune the machine learning model for enrollment prediction.
