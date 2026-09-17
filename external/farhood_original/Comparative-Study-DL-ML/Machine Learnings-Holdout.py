import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectFromModel
import random

# Establishing a random seed for reproducibility
random.seed(10)
np.random.seed(10)



# loading the dataset
try:

    data_math = pd.read_csv('student-dataset.csv')
except Exception as e:
    print("Reading the CSV file resulted in an error:")
    print(str(e))


data_math['Result'] = ['Pass' if x > 10 else 'Fail' for x in data_math['Grade']]
data_math = data_math.drop(columns='Grade', axis=1)

# Printing the number of columns, names of columns with the first few rows of the dataset
print(data_math.columns)
print(data_math.head())
print(data_math.shape[1])


# Identify columns that are not numeric
non_numerical_cols = data_math.select_dtypes(include=['object']).columns

# Encoding non-numerical columns with one-hot encoding
data_mathf = pd.get_dummies(data_math, columns=non_numerical_cols, drop_first=True)


# Printing the encoded Dataset
print(data_mathf)
print(data_mathf.shape[1])

# 'X' represents the input features, 'Y' denotes the objective variable
X = data_mathf.drop('Result_Pass', axis=1)
Y = data_mathf['Result_Pass']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# Machine learning models by Holdut

num_iterations = 100  # Number of iterations
accuracy_valuesLiR = []  # Reserved space for Linear Regression accuracy values
accuracy_valuesLiR_lasso = []  # Reserved space for Linear Regression with Lasso accuracy values
accuracy_valuesLoR = []  # Reserved space for Logistic Regression accuracy values
accuracy_valuesLoR_lasso = []  # Reserved space for Logistic Regression with Lasso accuracy values
accuracy_valuesSVM = []  # Reserved space for SVM accuracy values
accuracy_valuesSVM_lasso = []  # Reserved space for SVM with Lasso accuracy values
accuracy_valuesDT = []  # Reserved space for Decision Tree accuracy values
accuracy_valuesDT_lasso = []  # Reserved space for Decision Tree with Lasso accuracy values
accuracy_valuesRF = []  # Reserved space for Random Forest accuracy values
accuracy_valuesRF_lasso = []  # Reserved space for Random Forest with Lasso accuracy values
accuracy_valuesKNN = []  # Reserved space for KNN accuracy values
accuracy_valuesKNN_lasso = []  # Reserved space for KNN with Lasso accuracy values
accuracy_valuesXGB = [] # Reserved space for XGBoost wth Lasso accuracy values
accuracy_valuesXGB_lasso = [] # Reserved space for XGBoost without Lasso accuracy values


for iteration in range(num_iterations):
    # Splitting the data into training and testing sets using holdout validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=iteration)

    # Linear Regression without Lasso
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_lin = lin_reg.predict(X_test)
    accuracy_lin = accuracy_score(y_test, np.round(y_pred_lin))
    accuracy_valuesLiR.append(accuracy_lin)

    # Creating a list of potential alpha (Lasso penalty) values to test
    alphas = np.logspace(-2, -1, 10)

    # Initializing an array to store the cross-validated scores for each alpha
    cv_scores = []

    lr = LinearRegression()
    # Looping through the alphas
    for i in alphas:
        sel_ = SelectFromModel(Lasso(alpha=i, random_state=iteration))
        sel_.fit(X_train, y_train)
        X_train_selected = sel_.transform(X_train)

        # Fitting a linear regression model
        lr.fit(X_train_selected, y_train)

        # Performing cross-validation on the linear regression model
        scores = cross_val_score(lr, X_train_selected, y_train, cv=5, scoring='neg_mean_squared_error')

        # Storing the mean squared error (MSE) scores
        cv_scores.append(-np.mean(scores))

    # Finding the alpha with the best cross-validated score
    best_alpha = alphas[np.argmin(cv_scores)]

    # Printing the best amount of alpha
    print(best_alpha)


    # Definition of Lasso
    sel_ = SelectFromModel(Lasso(alpha=best_alpha, random_state=iteration))
    sel_.fit(X_train, y_train)
    X_train_selected = sel_.transform(X_train)
    X_test_selected = sel_.transform(X_test)


    # Linear Regression with Lasso
    lin_reg_with_lasso = LinearRegression()
    lin_reg_with_lasso.fit(X_train_selected, y_train)
    y_pred_lin_lasso = lin_reg_with_lasso.predict(X_test_selected)
    accuracy_lin_lasso = accuracy_score(y_test, np.round(y_pred_lin_lasso))
    accuracy_valuesLiR_lasso.append(accuracy_lin_lasso)

    # Logistic Regression without Lasso
    log_reg = LogisticRegression()
    log_reg.fit(X_train, y_train)
    y_pred_log = log_reg.predict(X_test)
    accuracy_log = accuracy_score(y_test, y_pred_log)
    accuracy_valuesLoR.append(accuracy_log)

    # Logistic Regression with Lasso
    log_reg_with_lasso = LogisticRegression()
    log_reg_with_lasso.fit(X_train_selected, y_train)
    y_pred_log_lasso = log_reg_with_lasso.predict(X_test_selected)
    accuracy_log_lasso = accuracy_score(y_test, y_pred_log_lasso)
    accuracy_valuesLoR_lasso.append(accuracy_log_lasso)

    # SVM without Lasso
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    accuracy_valuesSVM.append(accuracy_svm)


    # SVM with Lasso
    svm_with_lasso = SVC()
    svm_with_lasso.fit(X_train_selected, y_train)
    y_pred_svm_lasso = svm_with_lasso.predict(X_test_selected)
    accuracy_svm_lasso = accuracy_score(y_test, y_pred_svm_lasso)
    accuracy_valuesSVM_lasso.append(accuracy_svm_lasso)

    # Decision Tree without  Lasso
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    accuracy_valuesDT.append(accuracy_dt)

    # Decision Tree with Lasso
    dt_with_lasso = DecisionTreeClassifier()
    dt_with_lasso.fit(X_train_selected, y_train)
    y_pred_dt_lasso = dt_with_lasso.predict(X_test_selected)
    accuracy_dt_lasso = accuracy_score(y_test, y_pred_dt_lasso)
    accuracy_valuesDT_lasso.append(accuracy_dt_lasso)

    # Random Forest without Lasso
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    accuracy_valuesRF.append(accuracy_rf)

    # Random Forest with Lasso
    rf_with_lasso = RandomForestClassifier()
    rf_with_lasso.fit(X_train_selected, y_train)
    y_pred_rf_lasso = rf_with_lasso.predict(X_test_selected)
    accuracy_rf_lasso = accuracy_score(y_test, y_pred_rf_lasso)
    accuracy_valuesRF_lasso.append(accuracy_rf_lasso)

    # K-Nearest Neighbors without Lasso
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    accuracy_knn = accuracy_score(y_test, y_pred_knn)
    accuracy_valuesKNN.append(accuracy_knn)

    # KNN with Lasso
    knn_with_lasso = KNeighborsClassifier()
    knn_with_lasso.fit(X_train_selected, y_train)
    y_pred_knn_lasso = knn_with_lasso.predict(X_test_selected)
    accuracy_knn_lasso = accuracy_score(y_test, y_pred_knn_lasso)
    accuracy_valuesKNN_lasso.append(accuracy_knn_lasso)

    # XGBoost without Lasso
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    accuracy_valuesXGB.append(accuracy_xgb)

    # XGBoost with Lasso
    xgb_with_lasso = XGBClassifier()
    xgb_with_lasso.fit(X_train_selected, y_train)
    y_pred_xgb_lasso = xgb_with_lasso.predict(X_test_selected)
    accuracy_xgb_lasso = accuracy_score(y_test, y_pred_xgb_lasso)
    accuracy_valuesXGB_lasso.append(accuracy_xgb_lasso)



# Calculating mean accuracies for models without Lasso
mean_accuracyLiR = np.mean(accuracy_valuesLiR)
mean_accuracyLoR = np.mean(accuracy_valuesLoR)
mean_accuracySVM = np.mean(accuracy_valuesSVM)
mean_accuracyDT = np.mean(accuracy_valuesDT)
mean_accuracyRF = np.mean(accuracy_valuesRF)
mean_accuracyKNN = np.mean(accuracy_valuesKNN)
mean_accuracyXGB = np.mean(accuracy_valuesXGB)

# Calculating mean accuracies for models with Lasso
mean_accuracyLiR_lasso = np.mean(accuracy_valuesLiR_lasso)
mean_accuracyLoR_lasso = np.mean(accuracy_valuesLoR_lasso)
mean_accuracySVM_lasso = np.mean(accuracy_valuesSVM_lasso)
mean_accuracyDT_lasso = np.mean(accuracy_valuesDT_lasso)
mean_accuracyRF_lasso = np.mean(accuracy_valuesRF_lasso)
mean_accuracyKNN_lasso = np.mean(accuracy_valuesKNN_lasso)
mean_accuracyXGB_lasso = np.mean(accuracy_valuesXGB_lasso)

# Calculating mean accuracies as percentages for models without Lasso
mean_accuracyLiR_percent = mean_accuracyLiR * 100
mean_accuracyLoR_percent = mean_accuracyLoR * 100
mean_accuracySVM_percent = mean_accuracySVM * 100
mean_accuracyDT_percent = mean_accuracyDT * 100
mean_accuracyRF_percent = mean_accuracyRF * 100
mean_accuracyKNN_percent = mean_accuracyKNN * 100
mean_accuracyXGB_percent = mean_accuracyXGB * 100

# Calculating mean accuracies as percentages for models with Lasso
mean_accuracyLiR_lasso_percent = mean_accuracyLiR_lasso * 100
mean_accuracyLoR_lasso_percent = mean_accuracyLoR_lasso * 100
mean_accuracySVM_lasso_percent = mean_accuracySVM_lasso * 100
mean_accuracyDT_lasso_percent = mean_accuracyDT_lasso * 100
mean_accuracyRF_lasso_percent = mean_accuracyRF_lasso * 100
mean_accuracyKNN_lasso_percent = mean_accuracyKNN_lasso * 100
mean_accuracyXGB_lasso_percent = mean_accuracyXGB_lasso * 100

# Printing mean accuracies as percentages
print("Mean Accuracies Without Lasso:")
print(f"Linear Regression: {mean_accuracyLiR_percent:.2f}%")
print(f"Logistic Regression: {mean_accuracyLoR_percent:.2f}%")
print(f"SVM: {mean_accuracySVM_percent:.2f}%")
print(f"Decision Tree: {mean_accuracyDT_percent:.2f}%")
print(f"Random Forest: {mean_accuracyRF_percent:.2f}%")
print(f"KNN: {mean_accuracyKNN_percent:.2f}%")
print(f"XGBoost: {mean_accuracyXGB_percent:.2f}%")

print("\nMean Accuracies With Lasso:")
print(f"Linear Regression with Lasso: {mean_accuracyLiR_lasso_percent:.2f}%")
print(f"Logistic Regression with Lasso: {mean_accuracyLoR_lasso_percent:.2f}%")
print(f"SVM with Lasso: {mean_accuracySVM_lasso_percent:.2f}%")
print(f"Decision Tree with Lasso: {mean_accuracyDT_lasso_percent:.2f}%")
print(f"Random Forest with Lasso: {mean_accuracyRF_lasso_percent:.2f}%")
print(f"KNN with Lasso: {mean_accuracyKNN_lasso_percent:.2f}%")
print(f"XGBoost with Lasso: {mean_accuracyXGB_lasso_percent:.2f}%")


# Creating a boxplot for machine learning models with Lasso
plt.figure(figsize=(12, 6))
accuracy_values_with_lasso = [accuracy_valuesLiR_lasso, accuracy_valuesLoR_lasso, accuracy_valuesSVM_lasso,
                              accuracy_valuesDT_lasso, accuracy_valuesRF_lasso, accuracy_valuesKNN_lasso, accuracy_valuesXGB_lasso]
labels_with_lasso = ['Linear Regression with Lasso', 'Logistic Regression with Lasso', 'SVM with Lasso',
                     'Decision Tree with Lasso', 'Random Forest with Lasso', 'KNN with Lasso', 'XGBoost with Lasso']
plt.boxplot(accuracy_values_with_lasso, labels=labels_with_lasso, vert=False, showmeans=True)
plt.xlabel('Accuracy')
plt.title('Machine Learning with Lasso - Holdout', fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Creating a boxplot for machine learning models without Lasso
plt.figure(figsize=(12, 6))
accuracy_values_without_lasso = [accuracy_valuesLiR, accuracy_valuesLoR, accuracy_valuesSVM,
                                 accuracy_valuesDT, accuracy_valuesRF, accuracy_valuesKNN, accuracy_valuesXGB]
labels_without_lasso = ['Linear Regression', 'Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'KNN', 'XGBoost']
plt.boxplot(accuracy_values_without_lasso, labels=labels_without_lasso, vert=False, showmeans=True)
plt.xlabel('Accuracy')
plt.title('Machine Learning without Lasso - Holdout', fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
