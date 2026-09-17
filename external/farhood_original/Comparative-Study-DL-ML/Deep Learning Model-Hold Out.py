import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import gbnn
from sklearn.metrics import accuracy_score
from keras.layers import Dense
from bayes_opt import BayesianOptimization
import random

# Establishing a random seed for reproducibility
random.seed(10)
np.random.seed(10)
tf.random.set_seed(10)


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

# Identifying the features
num_features = X_scaled.shape[1]

# Deep learning models (NN-CNN-GBNN) by Holdut
# Reserving spaces for accuracy values
accuracy_values_nn_with_lasso = []
accuracy_values_nn_without_lasso = []
accuracy_values_cnn_with_lasso = []
accuracy_values_cnn_without_lasso = []
accuracy_values_gbnn_with_lasso = []
accuracy_values_gbnn_without_lasso = []

# Number of iterations
num_iterations = 100


for iteration in range(num_iterations):
    # Splitting the data into training and testing sets using holdout validation
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=iteration)



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



    # Defining the neural network model without Lasso
    def create_model_nn_without_lasso(units_layer1, units_layer2, dropout_rate, learning_rate):
        model_nn_without_lasso = Sequential()
        model_nn_without_lasso.add(Dense(units_layer1, input_dim=X_train.shape[1], activation='relu'))
        model_nn_without_lasso.add(Dense(units_layer2, activation='relu'))
        model_nn_without_lasso.add(Dense(1, activation='sigmoid'))
        model_nn_without_lasso.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        return model_nn_without_lasso


    def optimize_neural_network_nn_without_lasso(units_layer1, units_layer2, dropout_rate, learning_rate, batch_size, epochs):
        model_nn_without_lasso = create_model_nn_without_lasso(int(units_layer1), int(units_layer2), dropout_rate, learning_rate)
        model_nn_without_lasso.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)
        y_pred = model_nn_without_lasso.predict(X_test).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


    # Specifying the range for hyperparameter exploration
    pbounds = {
        'units_layer1': (5, 12),
        'units_layer2': (3, 8),
        'dropout_rate': (0.1, 0.5),
        'learning_rate': (0.0001, 0.1),
        'batch_size': (8, 128),
        'epochs': (100, 200)
    }

    # Defining the Bayesian optimization process
    optimizer = BayesianOptimization(
        f=optimize_neural_network_nn_without_lasso,
        pbounds=pbounds,
        random_state=0,
    )

    # Commencing the optimization process
    optimizer.maximize(init_points=2, n_iter=3)

    # Obtaining optimal hyperparameters
    best_hyperparameters = optimizer.max['params']
    best_units_layer1 = int(best_hyperparameters['units_layer1'])
    best_units_layer2 = int(best_hyperparameters['units_layer2'])
    best_dropout_rate = best_hyperparameters['dropout_rate']
    best_learning_rate = best_hyperparameters['learning_rate']
    best_batch_size = int(best_hyperparameters['batch_size'])
    best_epochs = int(best_hyperparameters['epochs'])


    # Training the ultimate model using the optimal hyperparameters
    final_model_nn_without_lasso = create_model_nn_without_lasso(best_units_layer1, best_units_layer2, best_dropout_rate, best_learning_rate)
    final_model_nn_without_lasso.fit(X_train, y_train, epochs=best_epochs, batch_size=best_batch_size, verbose=2)

    # Assessing the performance of the ultimate model
    y_pred = final_model_nn_without_lasso.predict(X_test).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred)
    accuracy_values_nn_without_lasso.append(test_accuracy * 100)





    # Defining the neural network model with Lasso
    def create_model_nn_with_lasso(units_layer1, units_layer2, dropout_rate, learning_rate):
        model_nn_with_lasso = Sequential()
        model_nn_with_lasso.add(Dense(units_layer1, input_dim=X_train_selected.shape[1], activation='relu'))
        model_nn_with_lasso.add(Dense(units_layer2, activation='relu'))
        model_nn_with_lasso.add(Dense(1, activation='sigmoid'))
        model_nn_with_lasso.compile(loss='binary_crossentropy',
                                       optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        return model_nn_with_lasso


    def optimize_neural_network_nn_with_lasso(units_layer1, units_layer2, dropout_rate, learning_rate, batch_size,
                                                 epochs):
        model_nn_with_lasso = create_model_nn_with_lasso(int(units_layer1), int(units_layer2), dropout_rate,
                                                               learning_rate)
        model_nn_with_lasso.fit(X_train_selected, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)
        y_pred = model_nn_with_lasso.predict(X_test_selected).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


    # Defining the Bayesian optimization process
    optimizer = BayesianOptimization(
        f=optimize_neural_network_nn_with_lasso,
        pbounds=pbounds,
        random_state=0,
    )

    # Commencing the optimization process
    optimizer.maximize(init_points=2, n_iter=3)

    # Obtaining optimal hyperparameters
    best_hyperparameters = optimizer.max['params']
    best_units_layer1 = int(best_hyperparameters['units_layer1'])
    best_units_layer2 = int(best_hyperparameters['units_layer2'])
    best_dropout_rate = best_hyperparameters['dropout_rate']
    best_learning_rate = best_hyperparameters['learning_rate']
    best_batch_size = int(best_hyperparameters['batch_size'])
    best_epochs = int(best_hyperparameters['epochs'])

    # Training the ultimate model using the optimal hyperparameters
    final_model_nn_with_lasso = create_model_nn_with_lasso(best_units_layer1, best_units_layer2,
                                                                 best_dropout_rate, best_learning_rate)
    final_model_nn_with_lasso.fit(X_train_selected, y_train, epochs=best_epochs, batch_size=best_batch_size, verbose=2)

    # Assessing the performance of the ultimate model
    y_pred = final_model_nn_with_lasso.predict(X_test_selected).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred)
    accuracy_values_nn_with_lasso.append(test_accuracy * 100)



    # Defining the CNN without Lasso neural network model
    def create_model_cnn_without_lasso(units_layer1, units_layer2, dropout_rate, learning_rate):
        model_cnn_without_lasso = Sequential()
        model_cnn_without_lasso.add(Conv1D(units_layer1, kernel_size=3, activation='relu', padding='same',
                                        input_shape=(X_train.shape[1], 1)))
        model_cnn_without_lasso.add(MaxPooling1D(pool_size=2))
        model_cnn_without_lasso.add(Flatten())
        model_cnn_without_lasso.add(Dense(units_layer2, activation='relu'))
        model_cnn_without_lasso.add(Dense(1, activation='sigmoid'))
        model_cnn_without_lasso.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
        return model_cnn_without_lasso


    def optimize_neural_network_cnn_without_lasso(units_layer1, units_layer2, dropout_rate, learning_rate, batch_size,
                                               epochs):
        model_cnn_without_lasso = create_model_cnn_without_lasso(int(units_layer1), int(units_layer2), dropout_rate,
                                                           learning_rate)
        model_cnn_without_lasso.fit(X_train, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)
        y_pred = model_cnn_without_lasso.predict(X_test).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


    # Defining the Bayesian optimization process
    optimizer = BayesianOptimization(
        f=optimize_neural_network_cnn_without_lasso,
        pbounds=pbounds,
        random_state=0,
    )

    # Commencing the optimization process
    optimizer.maximize(init_points=2, n_iter=3)

    # Obtaining optimal hyperparameters
    best_hyperparameters = optimizer.max['params']
    best_units_layer1 = int(best_hyperparameters['units_layer1'])
    best_units_layer2 = int(best_hyperparameters['units_layer2'])
    best_dropout_rate = best_hyperparameters['dropout_rate']
    best_learning_rate = best_hyperparameters['learning_rate']
    best_batch_size = int(best_hyperparameters['batch_size'])
    best_epochs = int(best_hyperparameters['epochs'])


    # Training the ultimate model using the optimal hyperparameters
    final_model_cnn_without_lasso = create_model_cnn_without_lasso(best_units_layer1, best_units_layer2,
                                                             best_dropout_rate, best_learning_rate)
    final_model_cnn_without_lasso.fit(X_train, y_train, epochs=best_epochs, batch_size=best_batch_size, verbose=2)

    # Assessing the performance of the ultimate model
    _, accuracy_cnn_without_lasso = final_model_cnn_without_lasso.evaluate(X_test, y_test, verbose=0)
    accuracy_values_cnn_without_lasso.append(accuracy_cnn_without_lasso * 100)





    # Defining the CNN with Lasso neural network model
    def create_model_cnn_with_lasso(units_layer1, units_layer2, dropout_rate, learning_rate):
        model_cnn_with_lasso = Sequential()
        model_cnn_with_lasso.add(Conv1D(units_layer1, kernel_size=3, activation='relu', padding='same',
                                           input_shape=(X_train_selected.shape[1], 1)))
        model_cnn_with_lasso.add(MaxPooling1D(pool_size=2))
        model_cnn_with_lasso.add(Flatten())
        model_cnn_with_lasso.add(Dense(units_layer2, activation='relu'))
        model_cnn_with_lasso.add(Dense(1, activation='sigmoid'))
        model_cnn_with_lasso.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                                        loss='binary_crossentropy', metrics=['accuracy'])
        return model_cnn_with_lasso



    def optimize_neural_network_cnn_with_lasso(units_layer1, units_layer2, dropout_rate, learning_rate, batch_size,
                                                 epochs):
        model_cnn_with_lasso = create_model_cnn_with_lasso(int(units_layer1), int(units_layer2), dropout_rate,
                                                               learning_rate)
        model_cnn_with_lasso.fit(X_train_selected, y_train, epochs=int(epochs), batch_size=int(batch_size), verbose=0)
        y_pred = model_cnn_with_lasso.predict(X_test_selected).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy


    # Defining the Bayesian optimization process
    optimizer = BayesianOptimization(
        f=optimize_neural_network_cnn_with_lasso,
        pbounds=pbounds,
        random_state=0,
    )

    # Commencing the optimization process
    optimizer.maximize(init_points=2, n_iter=3)

    # Obtaining optimal hyperparameters
    best_hyperparameters = optimizer.max['params']
    best_units_layer1 = int(best_hyperparameters['units_layer1'])
    best_units_layer2 = int(best_hyperparameters['units_layer2'])
    best_dropout_rate = best_hyperparameters['dropout_rate']
    best_learning_rate = best_hyperparameters['learning_rate']
    best_batch_size = int(best_hyperparameters['batch_size'])
    best_epochs = int(best_hyperparameters['epochs'])



    # Training the ultimate model using the optimal hyperparameters
    final_model_cnn_with_lasso = create_model_cnn_with_lasso(best_units_layer1, best_units_layer2,
                                                                 best_dropout_rate, best_learning_rate)
    final_model_cnn_with_lasso.fit(X_train_selected, y_train, epochs=best_epochs, batch_size=best_batch_size, verbose=2)

    # Assessing the performance of the ultimate model
    _, accuracy_cnn_with_lasso = final_model_cnn_with_lasso.evaluate(X_test_selected, y_test, verbose=0)
    accuracy_values_cnn_with_lasso.append(accuracy_cnn_with_lasso * 100)



    # Gradient Boosted Neural Network without Lasso feature selection

    # This section of code uses the GNEGNEClassifier from the GBNN package:
    # Authors: Emami, Seyedsaman and Martýnez-Muñoz, Gonzalo
    # Repository: "https://github.com/GAA-UAM/GBNN"
    # Paper: "https://ieeexplore.ieee.org/document/10110967"
    model_gbnn_without_lasso = gbnn.GNEGNEClassifier(total_nn=200, num_nn_step=1, eta=1.0, solver='adam',
                                                     subsample=0.5, tol=0.0, max_iter=200, random_state=None,
                                                     activation='logistic')
    model_gbnn_without_lasso.fit(X_train, y_train)
    y_pred_gbnn_without = model_gbnn_without_lasso.predict(X_test)
    accuracy_gbnn_without = accuracy_score(y_test, y_pred_gbnn_without)
    accuracy_values_gbnn_without_lasso.append(accuracy_gbnn_without * 100)

    
    # Gradient Boosted Neural Network with Lasso feature selection
    
    # This section of code uses the GNEGNEClassifier from the GBNN package:
    # Authors: Emami, Seyedsaman and Martýnez-Muñoz, Gonzalo
    # Repository: "https://github.com/GAA-UAM/GBNN"
    # Paper: "https://ieeexplore.ieee.org/document/10110967"
    model_gbnn_with_lasso = gbnn.GNEGNEClassifier(total_nn=200, num_nn_step=1, eta=1.0, solver='adam',
                                                  subsample=0.5, tol=0.0, max_iter=200, random_state=None,
                                                  activation='logistic')
    model_gbnn_with_lasso.fit(X_train_selected, y_train)
    y_pred_gbnn_without = model_gbnn_with_lasso.predict(X_test_selected)
    accuracy_gbnn_without = accuracy_score(y_test, y_pred_gbnn_without)
    accuracy_values_gbnn_with_lasso.append(accuracy_gbnn_without * 100)


# # Calculating mean accuracies values
mean_accuracy_nn_without_lasso = np.mean(accuracy_values_nn_without_lasso)
mean_accuracy_nn_with_lasso = np.mean(accuracy_values_nn_with_lasso)
mean_accuracy_cnn_without_lasso = np.mean(accuracy_values_cnn_without_lasso)
mean_accuracy_cnn_with_lasso = np.mean(accuracy_values_cnn_with_lasso)
mean_accuracy_gbnn_without_lasso = np.mean(accuracy_values_gbnn_without_lasso)
mean_accuracy_gbnn_with_lasso = np.mean(accuracy_values_gbnn_with_lasso)

# Printing mean accuracies
print("Mean Accuracy Values Without Lasso:")
print(f"NN without Lasso: {mean_accuracy_nn_without_lasso:.4f}")
print(f"NN with Lasso: {mean_accuracy_nn_with_lasso:.4f}")
print(f"CNN without Lasso: {mean_accuracy_cnn_without_lasso:.4f}")
print(f"CNN with Lasso: {mean_accuracy_cnn_with_lasso:.4f}")
print(f"GBNN without Lasso: {mean_accuracy_gbnn_without_lasso:.4f}")
print(f"GBNN with Lasso: {mean_accuracy_gbnn_with_lasso:.4f}")



# Generating a boxplot for visualization purposes
labels = ['NN without Lasso', 'NN with Lasso', 'CNN without Lasso', 'CNN with Lasso', 'GBNN without Lasso', 'GBNN with Lasso']

plt.figure(figsize=(12, 6))
plt.boxplot([accuracy_values_nn_without_lasso, accuracy_values_nn_with_lasso, accuracy_values_cnn_without_lasso, accuracy_values_cnn_with_lasso, accuracy_values_gbnn_without_lasso, accuracy_values_gbnn_with_lasso],
            labels=labels, vert=False, showmeans=True)
plt.xlabel('Accuracy')
plt.title('Deep Learning - Holdout',fontweight="bold")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

