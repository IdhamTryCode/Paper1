#!/usr/bin/env python3
"""
Comprehensive Classical ML Benchmark for Rust vs Python
Covers: Regression, Classification, Clustering
"""

import argparse
import json
import time
import psutil
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import make_classification, make_regression, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error, mean_absolute_error, r2_score,
    # Classification metrics
    accuracy_score, precision_score, recall_score, f1_score,
    # Clustering metrics
    silhouette_score, calinski_harabasz_score
)

# Regression algorithms
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

# Classification algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Clustering algorithms
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

def load_dataset(dataset_name, task_type, n_samples=1000, n_features=10):
    """Load or generate dataset based on task type"""
    np.random.seed(42)
    
    if task_type == "regression":
        if dataset_name == "synthetic":
            X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                                 noise=0.1, random_state=42)
        elif dataset_name == "california_housing":
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing()
            X, y = data.data, data.target
        else:
            raise ValueError(f"Unknown regression dataset: {dataset_name}")
            
    elif task_type == "classification":
        if dataset_name == "synthetic":
            X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                     n_classes=2, n_clusters_per_class=1,
                                     random_state=42)
        elif dataset_name == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            X, y = data.data, data.target
        else:
            raise ValueError(f"Unknown classification dataset: {dataset_name}")
            
    elif task_type == "clustering":
        if dataset_name == "synthetic":
            X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                             centers=3, random_state=42)
        elif dataset_name == "iris":
            from sklearn.datasets import load_iris
            data = load_iris()
            X, y = data.data, data.target
        else:
            raise ValueError(f"Unknown clustering dataset: {dataset_name}")
    
    # Split data for supervised learning
    if task_type in ["regression", "classification"]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        return X_train, X_test, y_train, y_test
    else:
        return X, y

def create_model(algorithm, task_type, **kwargs):
    """Create ML model based on algorithm and task type"""
    
    if task_type == "regression":
        models = {
            "linear": LinearRegression(),
            "ridge": Ridge(alpha=kwargs.get('alpha', 1.0)),
            "lasso": Lasso(alpha=kwargs.get('alpha', 1.0)),
            "elastic_net": ElasticNet(alpha=kwargs.get('alpha', 1.0)),
            "svr": SVR(kernel='rbf'),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "decision_tree": DecisionTreeRegressor(random_state=42)
        }
    elif task_type == "classification":
        models = {
            "logistic": LogisticRegression(random_state=42),
            "svm": SVC(kernel='rbf', random_state=42),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "decision_tree": DecisionTreeClassifier(random_state=42),
            "naive_bayes": GaussianNB()
        }
    elif task_type == "clustering":
        models = {
            "kmeans": KMeans(n_clusters=3, random_state=42),
            "dbscan": DBSCAN(eps=0.5, min_samples=5),
            "hierarchical": AgglomerativeClustering(n_clusters=3),
            "gaussian_mixture": GaussianMixture(n_components=3, random_state=42)
        }
    
    if algorithm not in models:
        raise ValueError(f"Unknown algorithm {algorithm} for task {task_type}")
    
    return models[algorithm]

def evaluate_model(model, X_train, X_test, y_train, y_test, task_type):
    """Evaluate model and return metrics"""
    
    # Record memory before training
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    # Train model
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    # Record peak memory
    memory_after = process.memory_info().rss / 1024 / 1024  # MB
    peak_memory = max(memory_before, memory_after)
    
    # Predict
    if task_type in ["regression", "classification"]:
        y_pred = model.predict(X_test)
    else:  # clustering
        y_pred = model.predict(X_test)
        y_test = y_test  # For clustering, we use original labels for evaluation
    
    # Calculate metrics
    metrics = {
        "training_time_seconds": training_time,
        "peak_memory_mb": peak_memory
    }
    
    if task_type == "regression":
        metrics.update({
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": mean_absolute_error(y_test, y_pred),
            "r2_score": r2_score(y_test, y_pred)
        })
    elif task_type == "classification":
        metrics.update({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted'),
            "recall": recall_score(y_test, y_pred, average='weighted'),
            "f1_score": f1_score(y_test, y_pred, average='weighted')
        })
    elif task_type == "clustering":
        # For clustering, we need to handle the case where DBSCAN might not assign all points
        if hasattr(model, 'labels_'):
            labels = model.labels_
        else:
            labels = y_pred
            
        # Only calculate metrics if we have valid clusters
        if len(np.unique(labels)) > 1:
            try:
                metrics.update({
                    "silhouette_score": silhouette_score(X_test, labels),
                    "calinski_harabasz_score": calinski_harabasz_score(X_test, labels)
                })
            except:
                metrics.update({
                    "silhouette_score": 0.0,
                    "calinski_harabasz_score": 0.0
                })
        else:
            metrics.update({
                "silhouette_score": 0.0,
                "calinski_harabasz_score": 0.0
            })
    
    return metrics

def run_benchmark(dataset, algorithm, task_type, alpha=1.0):
    """Run single benchmark"""
    
    print(f"Running {algorithm} on {dataset} ({task_type})")
    
    # Load dataset
    if task_type in ["regression", "classification"]:
        X_train, X_test, y_train, y_test = load_dataset(dataset, task_type)
    else:
        X, y = load_dataset(dataset, task_type)
        X_train, X_test, y_train, y_test = X, X, y, y  # For clustering
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = create_model(algorithm, task_type, alpha=alpha)
    
    # Evaluate model
    metrics = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, task_type)
    
    # Create result
    result = {
        "language": "python",
        "task_type": task_type,
        "algorithm": algorithm,
        "dataset": dataset,
        "alpha": alpha,
        "timestamp": datetime.now().isoformat(),
        **metrics
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Classical ML Benchmark")
    parser.add_argument("--dataset", default="synthetic", 
                       choices=["synthetic", "california_housing", "iris"])
    parser.add_argument("--algorithm", default="linear",
                       choices=["linear", "ridge", "lasso", "elastic_net", "svr", 
                               "random_forest", "decision_tree", "logistic", "svm",
                               "naive_bayes", "kmeans", "dbscan", "hierarchical", 
                               "gaussian_mixture"])
    parser.add_argument("--task_type", default="regression",
                       choices=["regression", "classification", "clustering"])
    parser.add_argument("--alpha", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Run benchmark
    result = run_benchmark(args.dataset, args.algorithm, args.task_type, args.alpha)
    
    # Save result
    with open("benchmark_results.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"Benchmark completed. Results saved to benchmark_results.json")
    print(f"Training time: {result['training_time_seconds']:.4f}s")
    print(f"Memory usage: {result['peak_memory_mb']:.2f} MB")
    
    # Print task-specific metrics
    if args.task_type == "regression":
        print(f"R² Score: {result['r2_score']:.4f}")
        print(f"RMSE: {result['rmse']:.4f}")
    elif args.task_type == "classification":
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"F1 Score: {result['f1_score']:.4f}")
    elif args.task_type == "clustering":
        print(f"Silhouette Score: {result['silhouette_score']:.4f}")

if __name__ == "__main__":
    main() 