import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
# pip install optuna-integration[mlflow]
from optuna.integration.mlflow import MLflowCallback

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import joblib
import time

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
    X, y, test_size=0.3, stratify=y
)

# Define Pipeline
pipeline = Pipeline(
    [
        ('Scaler', StandardScaler()),
        ('Model', KNeighborsClassifier())
    ]
)

# KNN Objective
def objective_knn(trial):
  # Define Hyperparameters
  scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
  pipeline.set_params(Scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
  pipeline.set_params(Model = KNeighborsClassifier())
  pipeline.set_params(Model__n_neighbors = trial.suggest_int('n_neighbors', 3, 21, 2))
  pipeline.set_params(Model__weights = trial.suggest_categorical('weights', ['uniform', 'distance']))
  pipeline.set_params(Model__p = trial.suggest_int('p', 1, 3))
  skf = StratifiedKFold(n_splits = 5, shuffle = True)
  score = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv = skf).mean()
  return score

# Decision Tree Objective
def objective_dt(trial):
  # Define hyperparameters
  scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
  pipeline.set_params(Scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler())
  pipeline.set_params(Model=DecisionTreeClassifier())
  pipeline.set_params(
      Model__criterion=trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
      Model__max_depth=trial.suggest_int('max_depth', 2, 30),
      Model__min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
      Model__min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
      Model__max_features=trial.suggest_categorical('max_features', [None, 'sqrt', 'log2']))
  skf = StratifiedKFold(n_splits=5, shuffle=True)
  score = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=skf).mean()
  return score

# SVM Objective
def objective_svm(trial):
    # Scaler (required for SVM)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(
        Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    )

    # Kernel first (controls valid params)
    kernel = trial.suggest_categorical(
        'kernel', ['linear', 'rbf', 'poly', 'sigmoid']
    )

    # Base hyperparameters
    params = {
        'C': trial.suggest_float('C', 1e-3, 1e2, log=True),
        'kernel': kernel
    }

    # Kernel-specific hyperparameters
    if kernel in ['rbf', 'poly', 'sigmoid']:
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e-1, log=True)

    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 5)

    # Set model
    pipeline.set_params(
        Model=SVC(**params)
    )
    skf = StratifiedKFold(n_splits=2, shuffle=True)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring='accuracy',
        cv=skf
    ).mean()
    return score

# GaussianNB Objective
def objective_gnb(trial):
    # Scaler (optional for GNB, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(
        Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    )

    # GaussianNB hyperparameters
    pipeline.set_params(
        Model=GaussianNB(
            var_smoothing=trial.suggest_float(
                'var_smoothing', 1e-11, 1e-7, log=True
            )
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring='accuracy',
        cv=skf
    ).mean()

    return score

# Random Forest Objective
def objective_rf(trial):
    # Scaler (optional for Random Forest, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(
        Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    )

    # Random Forest hyperparameters
    pipeline.set_params(
        Model=RandomForestClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
            criterion=trial.suggest_categorical(
                'criterion', ['gini', 'entropy', 'log_loss']
            ),
            max_depth=trial.suggest_int('max_depth', 5, 40),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', None]
            ),
            bootstrap=trial.suggest_categorical('bootstrap', [True, False]),
            random_state=42,
            n_jobs=-1
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring='accuracy',
        cv=skf
    ).mean()

    return score

# Gradient Boosting Objective
def objective_gb(trial):
    # Scaler (optional for Gradient Boosting, kept for consistency)
    scaler_type = trial.suggest_categorical('scaler_type', ['standard', 'minmax'])
    pipeline.set_params(
        Scaler=StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
    )

    # Gradient Boosting hyperparameters
    pipeline.set_params(
        Model=GradientBoostingClassifier(
            n_estimators=trial.suggest_int('n_estimators', 100, 500, step=50),
            learning_rate=trial.suggest_float(
                'learning_rate', 0.01, 0.3, log=True
            ),
            max_depth=trial.suggest_int('max_depth', 2, 10),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 20),
            max_features=trial.suggest_categorical(
                'max_features', ['sqrt', 'log2', None]
            ),
            subsample=trial.suggest_float('subsample', 0.5, 1.0),
            random_state=42
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    score = cross_val_score(
        pipeline,
        X_train,
        y_train,
        scoring='accuracy',
        cv=skf
    ).mean()

    return score

# Map model names to objective functions
objectives = {
    "KNN": objective_knn,
    "DecisionTree": objective_dt,
    "SVM": objective_svm,
    "GaussianNB": objective_gnb,
    "RandomForest": objective_rf,
    "GradientBoosting": objective_gb
}

# Set experiment
mlflow.set_experiment("IRIS_PL_RUNS")

results = {}
model_dict = {}
scaler_dict = {}

# Loop through each algorithm
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    # # Disable autologging once
    # mlflow.sklearn.autolog(disable = True)

    mlflow_cb = MLflowCallback(
        tracking_uri=None,              # Auto-detect (str/None)
        metric_name="cv_accuracy",      # Primary metric (str)
        mlflow_kwargs={                 # **MLflow start_run() kwargs**
            "nested": True              # Child runs under parent
        }
    )

    # Create Optuna study
    study = optuna.create_study(direction="maximize")

    # Train the final model
    start_fit = time.time()
    study.optimize(obj_fn, n_trials=20, callbacks=[mlflow_cb])
    fit_time = time.time() - start_fit

    print(f"Best CV accuracy for {model_name}: {study.best_value:.4f}")
    best_params = study.best_params
    results[model_name] = {"best_params": best_params, "best_cv_accuracy": study.best_value}

    # Fit the pipeline with the best parameters
    if model_name == "KNN":
        pipeline.set_params(
            Scaler=StandardScaler() if best_params["scaler_type"]=="standard" else MinMaxScaler(),
            Model__n_neighbors=best_params["n_neighbors"],
            Model__weights=best_params["weights"],
            Model__p=best_params["p"]
        )
    elif model_name == "DecisionTree":
        pipeline.set_params(
            Scaler=StandardScaler() if best_params["scaler_type"]=="standard" else MinMaxScaler(),
            Model__criterion=best_params["criterion"],
            Model__max_depth=best_params["max_depth"],
            Model__min_samples_split=best_params["min_samples_split"],
            Model__min_samples_leaf=best_params["min_samples_leaf"],
            Model__max_features=best_params["max_features"]
        )
    elif model_name == "SVM":
        scaler = StandardScaler() if best_params["scaler_type"]=="standard" else MinMaxScaler()
        pipeline.set_params(Scaler=scaler)
        params = {"kernel": best_params["kernel"], "C": best_params["C"]}
        if best_params["kernel"] in ["rbf", "poly", "sigmoid"]:
            params["gamma"] = best_params["gamma"]
        if best_params["kernel"] == "poly":
            params["degree"] = best_params["degree"]
        pipeline.set_params(Model=SVC(**params))
    elif model_name == "GaussianNB":
        pipeline.set_params(
            Scaler=StandardScaler() if best_params["scaler_type"]=="standard" else MinMaxScaler(),
            Model__var_smoothing=best_params["var_smoothing"]
        )
    elif model_name == "RandomForest":
        pipeline.set_params(
            Scaler=StandardScaler() if best_params["scaler_type"]=="standard" else MinMaxScaler(),
            Model__n_estimators=best_params["n_estimators"],
            Model__criterion=best_params["criterion"],
            Model__max_depth=best_params["max_depth"],
            Model__min_samples_split=best_params["min_samples_split"],
            Model__min_samples_leaf=best_params["min_samples_leaf"],
            Model__max_features=best_params["max_features"],
            Model__bootstrap=best_params["bootstrap"]
        )
    elif model_name == "GradientBoosting":
        pipeline.set_params(
            Scaler=StandardScaler() if best_params["scaler_type"]=="standard" else MinMaxScaler(),
            Model__n_estimators=best_params["n_estimators"],
            Model__learning_rate=best_params["learning_rate"],
            Model__max_depth=best_params["max_depth"],
            Model__min_samples_split=best_params["min_samples_split"],
            Model__min_samples_leaf=best_params["min_samples_leaf"],
            Model__max_features=best_params["max_features"],
            Model__subsample=best_params["subsample"]
        )

    # # Enable autologging once
    # mlflow.sklearn.autolog()

    # Train the final model
    pipeline.fit(X_train, y_train)

    # Evaluate on test data
    start_test = time.time()
    y_pred = pipeline.predict(X_test)
    test_time = time.time() - start_test

    train_acc = pipeline.score(X_train, y_train)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"{model_name} Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}")
    print(f"{model_name} Fit Time: {fit_time:.2f}s, Test Time: {test_time:.2f}s")

    # Save model manually to track model size
    model_path = f"{model_name}_final_model.pkl"
    joblib.dump(pipeline, model_path)
    model_size = os.path.getsize(model_path)
    for i, model in enumerate(objectives.keys()):
        model_dict[model] = i

    for i, scaler_type in enumerate(['standard', 'minmax']):
        scaler_dict[scaler_type] = i

    mlflow.log_metric(f"model_id", model_dict[model_name])
    mlflow.log_metric(f"Scalar_id", scaler_dict[scaler_type])
    mlflow.log_metric(f"train_accuracy", train_acc)
    mlflow.log_metric(f"test_accuracy", test_acc)
    mlflow.log_metric(f"train_time", fit_time)
    mlflow.log_metric(f"test_time", test_time)
    mlflow.log_metric(f"model_size", model_size)
    mlflow.sklearn.log_model(pipeline, name=f"{model_name}iris_model") #, registered_model_name = f"Iris_{model_name}_Best")
    os.remove(model_path)

    results[model_name].update({
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "fit_time": fit_time,
        "test_time": test_time,
        "model_size_bytes": model_size
    })
    mlflow.end_run()

# Summary
print("\n--- Summary ---")
for model_name, res in results.items():
    print(f"{model_name}: CV Acc={res['best_cv_accuracy']:.4f}, Train Acc={res['train_accuracy']:.4f}, "
          f"Test Acc={res['test_accuracy']:.4f}, Fit Time={res['fit_time']:.2f}s, "
          f"Model Size={res['model_size_bytes']} bytes")