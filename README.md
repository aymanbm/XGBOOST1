# Heart Disease Prediction
This project aims to predict the presence of heart disease in patients using various machine learning models. The goal is to compare the performance of XGBoost, Support Vector Machine (SVM), and Artificial Neural Networks (ANN) for accurately predicting heart disease based on patient data.

## Objective
The main objective of this project is to predict the likelihood of heart disease in patients using several key medical attributes. The project evaluates and compares the performance of three machine learning models: XGBoost, SVM, and ANN, with a focus on optimizing the XGBoost model using Random Search and improving the ANN model with techniques like Dropout, Glorot Uniform Initialization, and L1/L2 Regularization.

## Dataset Overview
The dataset used in this project includes various medical attributes that influence the likelihood of heart disease. The dataset features the following attributes:

Age: The age of the patient.

Sex: Gender of the patient (Male/Female).

Chest Pain Type: Type of chest pain experienced (categorical).

Resting Blood Pressure (s): Resting blood pressure of the patient.

Cholesterol: Serum cholesterol level in mg/dl.

Fasting Blood Sugar: Whether the patient’s fasting blood sugar is greater than 120 mg/dl (binary: 1 = true, 0 = false).

Resting Electrocardiographic Results (ECG): Electrocardiographic results (categorical).

Max Heart Rate: Maximum heart rate achieved during exercise.

Exercise Angina: Whether the patient has exercise-induced angina (binary: 1 = yes, 0 = no).

Oldpeak: Depression induced by exercise relative to rest.

ST Slope: The slope of the peak exercise ST segment (categorical).

Target: Whether the patient has heart disease (binary: 1 = yes, 0 = no).

## Data Preprocessing
Data Cleaning: Handled missing values, outliers, and ensured correct data types for each feature.

Feature Engineering: Converted categorical features like Chest Pain Type, Resting ECG, and ST Slope into numerical representations using one-hot encoding.

Scaling: Standardized numerical features (e.g., Age, Cholesterol, Max Heart Rate) for models sensitive to feature scaling, like SVM and ANN.

Split the Data: Divided the dataset into training and testing sets for model evaluation.

## Models Compared
1. XGBoost
XGBoost was used due to its high performance with tabular data and ability to handle missing values efficiently.

The model was optimized using Random Search to find the best hyperparameters, improving the model’s accuracy and robustness.

2. Support Vector Machine (SVM)
SVM with an RBF kernel was applied to separate the data points in higher-dimensional space.

Hyperparameters such as the C parameter and gamma were fine-tuned to improve performance.

3. Artificial Neural Network (ANN)
The ANN was designed with multiple hidden layers to capture complex non-linear relationships in the data.

## Key techniques to improve performance included:

Dropout: Used to reduce overfitting by randomly deactivating some neurons during training.

Glorot Uniform Initialization: Used to initialize the network’s weights to avoid issues with vanishing or exploding gradients.

L1/L2 Regularization: Applied to control the magnitude of the weights and prevent overfitting.

Adamax Optimizer: Used as the optimizer, which is more stable when dealing with sparse gradients.

## Model Evaluation
Models were evaluated using the following metrics:

Accuracy

Precision 

MSE

Results and Discussion
XGBoost showed strong performance, handling non-linear relationships well and providing feature importance insights.

SVM was effective at distinguishing between classes, especially when the data was properly scaled, but was slower for larger datasets.

ANN demonstrated competitive results with proper regularization and optimization, though it required more computational resources compared to the other models.

| Variable       | Accuracy | Precision | MSE |
|----------------|--------------------|----------------|----------------|
| `SVM`         | 0.8                  | 0.846          | 0.2         |
| `XGBOOST`| 0.842         | 0.875            | 0.157  |
| `ANN`  | 0.8586          | 0.9      | 0.1413     |


## Installation & Setup

Ensure you have Python installed along with the required libraries.

Install dependencies using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
