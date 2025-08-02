#!/usr/bin/env python3
"""
Simple Python Regression Benchmark for Testing
"""

import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing, make_regression
from sklearn.preprocessing import StandardScaler
import psutil
import argparse
import json
from datetime import datetime

def load_dataset(dataset_name):
    """Load dataset"""
    if dataset_name == "california_housing":
        data = fetch_california_housing()
        X, y = data.data, data.target
    elif dataset_name == "synthetic":
        X, y = make_regression(n_samples=1000, n_features=20, n_informative=10, 
                             random_state=42, noise=0.1)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"Loaded dataset {dataset_name}: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y

def create_model(algorithm, **kwargs):
    """Create model"""
    if algorithm == "linear":
        return LinearRegression()
    elif algorithm == "ridge":
        return Ridge(alpha=kwargs.get("alpha", 1.0), random_state=42)
    elif algorithm == "lasso":
        return Lasso(alpha=kwargs.get("alpha", 1.0), random_state=42)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

def benchmark_model(dataset_name, algorithm, **kwargs):
    """Run benchmark"""
    print(f"Running benchmark: {algorithm} on {dataset_name}")
    
    # Load data
    X, y = load_dataset(dataset_name)
    
    # Preprocess
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Create model
    model = create_model(algorithm, **kwargs)
    
    # Monitor resources
    process = psutil.Process()
    start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    start_time = time.perf_counter()
    
    # Train
    model.fit(X_train, y_train)
    
    # Get metrics
    training_time = time.perf_counter() - start_time
    end_memory = process.memory_info().rss / (1024 * 1024)  # MB
    peak_memory = end_memory - start_memory
    
    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Results
    results = {
        "dataset": dataset_name,
        "algorithm": algorithm,
        "training_time_seconds": training_time,
        "peak_memory_mb": peak_memory,
        "rmse": rmse,
        "mae": mae,
        "r2_score": r2,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"Results: {json.dumps(results, indent=2)}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Simple Regression Benchmark")
    parser.add_argument("--dataset", default="california_housing", 
                       choices=["california_housing", "synthetic"])
    parser.add_argument("--algorithm", default="linear", 
                       choices=["linear", "ridge", "lasso"])
    parser.add_argument("--alpha", type=float, default=1.0)
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_model(
        dataset_name=args.dataset,
        algorithm=args.algorithm,
        alpha=args.alpha
    )
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("Benchmark completed! Results saved to benchmark_results.json")

if __name__ == "__main__":
    main() 