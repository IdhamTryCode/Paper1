#!/usr/bin/env python3
"""
Run Multiple Benchmarks and Compare Results
"""

import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def run_benchmark(dataset, algorithm, alpha=1.0):
    """Run a single benchmark"""
    cmd = [
        "python", "simple_benchmark.py",
        "--dataset", dataset,
        "--algorithm", algorithm,
        "--alpha", str(alpha)
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        # Load results from file
        with open("benchmark_results.json", "r") as f:
            return json.load(f)
    else:
        print(f"Error running benchmark: {result.stderr}")
        return None

def run_comprehensive_benchmarks():
    """Run comprehensive benchmarks"""
    datasets = ["california_housing", "synthetic"]
    algorithms = [
        ("linear", {}),
        ("ridge", {"alpha": 0.1}),
        ("ridge", {"alpha": 1.0}),
        ("ridge", {"alpha": 10.0}),
        ("lasso", {"alpha": 0.1}),
        ("lasso", {"alpha": 1.0}),
        ("lasso", {"alpha": 10.0})
    ]
    
    all_results = []
    
    for dataset in datasets:
        for algorithm, params in algorithms:
            alpha = params.get("alpha", 1.0)
            result = run_benchmark(dataset, algorithm, alpha)
            if result:
                all_results.append(result)
                print(f"✓ Completed: {algorithm} on {dataset}")
            else:
                print(f"✗ Failed: {algorithm} on {dataset}")
    
    return all_results

def analyze_results(results):
    """Analyze and visualize results"""
    if not results:
        print("No results to analyze")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    
    print("\nPerformance Metrics:")
    print(df[['algorithm', 'dataset', 'training_time_seconds', 'peak_memory_mb', 'r2_score']].to_string(index=False))
    
    print("\nBest Performance by Algorithm:")
    best_by_algorithm = df.loc[df.groupby('algorithm')['r2_score'].idxmax()]
    print(best_by_algorithm[['algorithm', 'dataset', 'r2_score', 'training_time_seconds']].to_string(index=False))
    
    print("\nFastest Training:")
    fastest = df.loc[df['training_time_seconds'].idxmin()]
    print(f"Algorithm: {fastest['algorithm']}, Dataset: {fastest['dataset']}, Time: {fastest['training_time_seconds']:.4f}s")
    
    print("\nBest R² Score:")
    best_r2 = df.loc[df['r2_score'].idxmax()]
    print(f"Algorithm: {best_r2['algorithm']}, Dataset: {best_r2['dataset']}, R²: {best_r2['r2_score']:.4f}")
    
    # Create visualizations
    create_visualizations(df)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"comprehensive_results_{timestamp}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")

def create_visualizations(df):
    """Create visualization plots"""
    # Set style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Training Time Comparison
    ax1 = axes[0, 0]
    sns.boxplot(data=df, x='algorithm', y='training_time_seconds', ax=ax1)
    ax1.set_title('Training Time by Algorithm')
    ax1.set_ylabel('Training Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Memory Usage Comparison
    ax2 = axes[0, 1]
    sns.boxplot(data=df, x='algorithm', y='peak_memory_mb', ax=ax2)
    ax2.set_title('Memory Usage by Algorithm')
    ax2.set_ylabel('Peak Memory (MB)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. R² Score Comparison
    ax3 = axes[1, 0]
    sns.boxplot(data=df, x='algorithm', y='r2_score', ax=ax3)
    ax3.set_title('R² Score by Algorithm')
    ax3.set_ylabel('R² Score')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. RMSE Comparison
    ax4 = axes[1, 1]
    sns.boxplot(data=df, x='algorithm', y='rmse', ax=ax4)
    ax4.set_title('RMSE by Algorithm')
    ax4.set_ylabel('RMSE')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_file = f"benchmark_visualization_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {plot_file}")
    
    # Show plot
    plt.show()

def main():
    print("🚀 Starting Comprehensive ML Benchmark Suite")
    print("="*50)
    
    # Run benchmarks
    results = run_comprehensive_benchmarks()
    
    # Analyze results
    analyze_results(results)
    
    print("\n✅ Benchmark suite completed!")

if __name__ == "__main__":
    main() 