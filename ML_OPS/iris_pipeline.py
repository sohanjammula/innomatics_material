import numpy as np
import pandas as pd
import optuna

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # or 1

import warnings
warnings.filterwarnings("ignore")

# Load Dataset
iris = load_iris(as_frame=True)
features = iris.data
label = iris.target
feature_names = iris.feature_names

data = pd.DataFrame(
    np.hstack((features, label.values.reshape(-1, 1))),
    columns=feature_names + ["target"],
)

# Data Cleaning
data = data.drop_duplicates()

# Segregate features and Target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Baseline Model with MLflow
scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=75)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn.fit(X_train_scaled, y_train)

train_acc = knn.score(X_train_scaled, y_train)
test_pred = knn.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_pred)

# print('Training accuracy', train_acc)
# print('Testing accuracy', test_acc)


# Build a pipeline
pipeline_1 = Pipeline(
    [
        ('Scaler', StandardScaler()),
        ('Model', KNeighborsClassifier())
    ]
)

# Create Objective
def objective(trial):
  # Define Hyperparameters
  scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
  pipeline_1.set_params(Scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
  pipeline_1.set_params(Model__n_neighbors = trial.suggest_int('n_neighbors', 3, 21, 2))
  pipeline_1.set_params(Model__p = trial.suggest_int('p', 1, 3))
  skf = StratifiedKFold(n_splits = 5, shuffle = True)
  score = cross_val_score(pipeline_1, X_train, y_train, scoring='accuracy', cv = skf).mean()
  return score

# Define a study
study = optuna.create_study(direction = 'maximize')

# Run the study
study.optimize(objective, n_trials=100)

# Best parameters
best_params = study.best_params
print('Best Parameters\n', best_params)
print('Best accuracy', study.best_value)

# Training with Best parameters
pipeline_1.set_params(Scaler = StandardScaler())
pipeline_1.set_params(**{'Model__n_neighbors': best_params['n_neighbors'], 'Model__p': best_params['p']})
pipeline_1.fit(X_train, y_train)
score = pipeline_1.score(X_train, y_train)
print('Training score', score)

# Testing the model
y_pred_test = pipeline_1.predict(X_test)
test_score = accuracy_score(y_test, y_pred_test)
print('Testing score', test_score)