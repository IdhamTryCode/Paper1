#!/usr/bin/env python3
"""
Generate Comprehensive Metrics Visualization for Classical ML Benchmark Results
Creates tables, charts, and performance comparisons
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_results():
    """Load benchmark results"""
    with open("results/classical_ml_benchmark_results.json", "r") as f:
        return json.load(f)

def create_performance_table(df):
    """Create detailed performance comparison table"""
    
    # Group by language and task type
    summary = df.groupby(['language', 'task_type']).agg({
        'training_time_seconds': ['mean', 'std', 'min', 'max'],
        'peak_memory_mb': ['mean', 'std', 'min', 'max'],
        'algorithm': 'count'
    }).round(4)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Add performance metrics
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        
        if task_type == "regression":
            for lang in task_df['language'].unique():
                mask = (summary['language'] == lang) & (summary['task_type'] == task_type)
                lang_task_df = task_df[task_df['language'] == lang]
                summary.loc[mask, 'avg_r2_score'] = lang_task_df['r2_score'].mean()
                summary.loc[mask, 'avg_rmse'] = lang_task_df['rmse'].mean()
                
        elif task_type == "classification":
            for lang in task_df['language'].unique():
                mask = (summary['language'] == lang) & (summary['task_type'] == task_type)
                lang_task_df = task_df[task_df['language'] == lang]
                summary.loc[mask, 'avg_accuracy'] = lang_task_df['accuracy'].mean()
                summary.loc[mask, 'avg_f1_score'] = lang_task_df['f1_score'].mean()
                
        elif task_type == "clustering":
            for lang in task_df['language'].unique():
                mask = (summary['language'] == lang) & (summary['task_type'] == task_type)
                lang_task_df = task_df[task_df['language'] == lang]
                summary.loc[mask, 'avg_silhouette'] = lang_task_df['silhouette_score'].mean()
    
    return summary

def create_algorithm_comparison_table(df):
    """Create algorithm-specific comparison table"""
    
    # Get best performing algorithm for each task type and language
    best_algorithms = []
    
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        
        for lang in task_df['language'].unique():
            lang_df = task_df[task_df['language'] == lang]
            
            # Best by training time
            fastest = lang_df.loc[lang_df['training_time_seconds'].idxmin()]
            best_algorithms.append({
                'task_type': task_type,
                'language': lang,
                'metric': 'Fastest Training',
                'algorithm': fastest['algorithm'],
                'dataset': fastest['dataset'],
                'value': fastest['training_time_seconds'],
                'unit': 'seconds'
            })
            
            # Best by memory efficiency
            most_efficient = lang_df.loc[lang_df['peak_memory_mb'].idxmin()]
            best_algorithms.append({
                'task_type': task_type,
                'language': lang,
                'metric': 'Most Memory Efficient',
                'algorithm': most_efficient['algorithm'],
                'dataset': most_efficient['dataset'],
                'value': most_efficient['peak_memory_mb'],
                'unit': 'MB'
            })
            
            # Best by accuracy/quality
            if task_type == "regression":
                best_quality = lang_df.loc[lang_df['r2_score'].idxmax()]
                best_algorithms.append({
                    'task_type': task_type,
                    'language': lang,
                    'metric': 'Best R² Score',
                    'algorithm': best_quality['algorithm'],
                    'dataset': best_quality['dataset'],
                    'value': best_quality['r2_score'],
                    'unit': 'R²'
                })
            elif task_type == "classification":
                best_quality = lang_df.loc[lang_df['accuracy'].idxmax()]
                best_algorithms.append({
                    'task_type': task_type,
                    'language': lang,
                    'metric': 'Best Accuracy',
                    'algorithm': best_quality['algorithm'],
                    'dataset': best_quality['dataset'],
                    'value': best_quality['accuracy'],
                    'unit': 'Accuracy'
                })
            elif task_type == "clustering":
                best_quality = lang_df.loc[lang_df['silhouette_score'].idxmax()]
                best_algorithms.append({
                    'task_type': task_type,
                    'language': lang,
                    'metric': 'Best Silhouette Score',
                    'algorithm': best_quality['algorithm'],
                    'dataset': best_quality['dataset'],
                    'value': best_quality['silhouette_score'],
                    'unit': 'Silhouette'
                })
    
    return pd.DataFrame(best_algorithms)

def create_visualizations(df):
    """Create comprehensive visualizations"""
    
    # Set up the plotting
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Performance Comparison by Task Type
    plt.subplot(4, 3, 1)
    task_lang_means = df.groupby(['task_type', 'language'])['training_time_seconds'].mean().unstack()
    task_lang_means.plot(kind='bar', ax=plt.gca())
    plt.title('Average Training Time by Task Type')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.legend(title='Language')
    
    # 2. Memory Usage Comparison
    plt.subplot(4, 3, 2)
    task_lang_memory = df.groupby(['task_type', 'language'])['peak_memory_mb'].mean().unstack()
    task_lang_memory.plot(kind='bar', ax=plt.gca())
    plt.title('Average Memory Usage by Task Type')
    plt.ylabel('Memory Usage (MB)')
    plt.xticks(rotation=45)
    plt.legend(title='Language')
    
    # 3. Quality Metrics Comparison
    plt.subplot(4, 3, 3)
    quality_metrics = []
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        for lang in task_df['language'].unique():
            lang_df = task_df[task_df['language'] == lang]
            if task_type == "regression":
                quality_metrics.append({
                    'task_type': task_type,
                    'language': lang,
                    'quality': lang_df['r2_score'].mean(),
                    'metric': 'R² Score'
                })
            elif task_type == "classification":
                quality_metrics.append({
                    'task_type': task_type,
                    'language': lang,
                    'quality': lang_df['accuracy'].mean(),
                    'metric': 'Accuracy'
                })
            elif task_type == "clustering":
                quality_metrics.append({
                    'task_type': task_type,
                    'language': lang,
                    'quality': lang_df['silhouette_score'].mean(),
                    'metric': 'Silhouette Score'
                })
    
    quality_df = pd.DataFrame(quality_metrics)
    quality_pivot = quality_df.pivot(index='task_type', columns='language', values='quality')
    quality_pivot.plot(kind='bar', ax=plt.gca())
    plt.title('Average Quality Metrics by Task Type')
    plt.ylabel('Quality Score')
    plt.xticks(rotation=45)
    plt.legend(title='Language')
    
    # 4. Algorithm Performance Heatmap
    plt.subplot(4, 3, 4)
    algo_perf = df.groupby(['algorithm', 'language'])['training_time_seconds'].mean().unstack()
    sns.heatmap(algo_perf, annot=True, fmt='.3f', cmap='YlOrRd', ax=plt.gca())
    plt.title('Algorithm Training Time Heatmap')
    plt.ylabel('Algorithm')
    
    # 5. Memory Usage Heatmap
    plt.subplot(4, 3, 5)
    algo_memory = df.groupby(['algorithm', 'language'])['peak_memory_mb'].mean().unstack()
    sns.heatmap(algo_memory, annot=True, fmt='.1f', cmap='Blues', ax=plt.gca())
    plt.title('Algorithm Memory Usage Heatmap')
    plt.ylabel('Algorithm')
    
    # 6. Performance vs Memory Scatter
    plt.subplot(4, 3, 6)
    for lang in df['language'].unique():
        lang_df = df[df['language'] == lang]
        plt.scatter(lang_df['peak_memory_mb'], lang_df['training_time_seconds'], 
                   label=lang, alpha=0.7, s=50)
    plt.xlabel('Memory Usage (MB)')
    plt.ylabel('Training Time (seconds)')
    plt.title('Performance vs Memory Usage')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    
    # 7. Algorithm Distribution by Task Type
    plt.subplot(4, 3, 7)
    algo_counts = df.groupby(['task_type', 'algorithm']).size().unstack(fill_value=0)
    algo_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Algorithm Distribution by Task Type')
    plt.ylabel('Number of Benchmarks')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 8. Performance Improvement Percentage
    plt.subplot(4, 3, 8)
    improvements = []
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        python_time = task_df[task_df['language'] == 'python']['training_time_seconds'].mean()
        rust_time = task_df[task_df['language'] == 'rust']['training_time_seconds'].mean()
        improvement = ((python_time - rust_time) / python_time) * 100
        improvements.append({'task_type': task_type, 'improvement': improvement})
    
    improvement_df = pd.DataFrame(improvements)
    plt.bar(improvement_df['task_type'], improvement_df['improvement'], color='green')
    plt.title('Rust Performance Improvement (%)')
    plt.ylabel('Improvement (%)')
    plt.xticks(rotation=45)
    
    # 9. Memory Efficiency Improvement
    plt.subplot(4, 3, 9)
    memory_improvements = []
    for task_type in df['task_type'].unique():
        task_df = df[df['task_type'] == task_type]
        python_memory = task_df[task_df['language'] == 'python']['peak_memory_mb'].mean()
        rust_memory = task_df[task_df['language'] == 'rust']['peak_memory_mb'].mean()
        improvement = ((python_memory - rust_memory) / python_memory) * 100
        memory_improvements.append({'task_type': task_type, 'improvement': improvement})
    
    memory_improvement_df = pd.DataFrame(memory_improvements)
    plt.bar(memory_improvement_df['task_type'], memory_improvement_df['improvement'], color='blue')
    plt.title('Rust Memory Efficiency Improvement (%)')
    plt.ylabel('Improvement (%)')
    plt.xticks(rotation=45)
    
    # 10. Quality Comparison by Algorithm
    plt.subplot(4, 3, 10)
    quality_by_algo = []
    for algorithm in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algorithm]
        for lang in algo_df['language'].unique():
            lang_df = algo_df[algo_df['language'] == lang]
            if 'r2_score' in lang_df.columns:
                quality_by_algo.append({
                    'algorithm': algorithm,
                    'language': lang,
                    'quality': lang_df['r2_score'].mean(),
                    'metric': 'R² Score'
                })
            elif 'accuracy' in lang_df.columns:
                quality_by_algo.append({
                    'algorithm': algorithm,
                    'language': lang,
                    'quality': lang_df['accuracy'].mean(),
                    'metric': 'Accuracy'
                })
            elif 'silhouette_score' in lang_df.columns:
                quality_by_algo.append({
                    'algorithm': algorithm,
                    'language': lang,
                    'quality': lang_df['silhouette_score'].mean(),
                    'metric': 'Silhouette Score'
                })
    
    quality_algo_df = pd.DataFrame(quality_by_algo)
    if not quality_algo_df.empty:
        quality_algo_pivot = quality_algo_df.pivot(index='algorithm', columns='language', values='quality')
        quality_algo_pivot.plot(kind='bar', ax=plt.gca())
        plt.title('Quality by Algorithm')
        plt.ylabel('Quality Score')
        plt.xticks(rotation=45)
        plt.legend(title='Language')
    
    # 11. Dataset Performance Comparison
    plt.subplot(4, 3, 11)
    dataset_perf = df.groupby(['dataset', 'language'])['training_time_seconds'].mean().unstack()
    dataset_perf.plot(kind='bar', ax=plt.gca())
    plt.title('Performance by Dataset')
    plt.ylabel('Training Time (seconds)')
    plt.xticks(rotation=45)
    plt.legend(title='Language')
    
    # 12. Summary Statistics
    plt.subplot(4, 3, 12)
    plt.axis('off')
    summary_text = f"""
    BENCHMARK SUMMARY
    =================
    Total Benchmarks: {len(df)}
    Languages: {', '.join(df['language'].unique())}
    Task Types: {', '.join(df['task_type'].unique())}
    Algorithms: {len(df['algorithm'].unique())}
    Datasets: {', '.join(df['dataset'].unique())}
    
    AVERAGE PERFORMANCE
    ===================
    Python Training Time: {df[df['language']=='python']['training_time_seconds'].mean():.4f}s
    Rust Training Time: {df[df['language']=='rust']['training_time_seconds'].mean():.4f}s
    Python Memory: {df[df['language']=='python']['peak_memory_mb'].mean():.1f} MB
    Rust Memory: {df[df['language']=='rust']['peak_memory_mb'].mean():.1f} MB
    
    RUST IMPROVEMENTS
    ================
    Training Time: {((df[df['language']=='python']['training_time_seconds'].mean() - df[df['language']=='rust']['training_time_seconds'].mean()) / df[df['language']=='python']['training_time_seconds'].mean() * 100):.1f}%
    Memory Usage: {((df[df['language']=='python']['peak_memory_mb'].mean() - df[df['language']=='rust']['peak_memory_mb'].mean()) / df[df['language']=='python']['peak_memory_mb'].mean() * 100):.1f}%
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    return fig

def save_tables(performance_table, algorithm_table):
    """Save tables to files"""
    
    # Save performance table
    performance_table.to_csv("results/performance_summary_table.csv", index=False)
    performance_table.to_html("results/performance_summary_table.html", index=False)
    
    # Save algorithm comparison table
    algorithm_table.to_csv("results/algorithm_comparison_table.csv", index=False)
    algorithm_table.to_html("results/algorithm_comparison_table.html", index=False)
    
    # Create markdown table
    with open("results/performance_summary_table.md", "w") as f:
        f.write("# Performance Summary Table\n\n")
        f.write(performance_table.to_markdown(index=False))
    
    with open("results/algorithm_comparison_table.md", "w") as f:
        f.write("# Algorithm Comparison Table\n\n")
        f.write(algorithm_table.to_markdown(index=False))

def main():
    print("📊 Generating Comprehensive Metrics Visualization...")
    
    # Load data
    results = load_results()
    df = pd.DataFrame(results)
    
    print(f"Loaded {len(df)} benchmark results")
    
    # Create tables
    print("Creating performance tables...")
    performance_table = create_performance_table(df)
    algorithm_table = create_algorithm_comparison_table(df)
    
    # Save tables
    print("Saving tables...")
    save_tables(performance_table, algorithm_table)
    
    # Create visualizations
    print("Creating visualizations...")
    fig = create_visualizations(df)
    
    # Save visualization
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fig.savefig(f"results/comprehensive_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"results/comprehensive_metrics_{timestamp}.pdf", bbox_inches='tight')
    
    print("\n✅ Generated Files:")
    print(f"📄 Tables:")
    print(f"   - results/performance_summary_table.csv")
    print(f"   - results/performance_summary_table.html")
    print(f"   - results/performance_summary_table.md")
    print(f"   - results/algorithm_comparison_table.csv")
    print(f"   - results/algorithm_comparison_table.html")
    print(f"   - results/algorithm_comparison_table.md")
    print(f"📊 Visualizations:")
    print(f"   - results/comprehensive_metrics_{timestamp}.png")
    print(f"   - results/comprehensive_metrics_{timestamp}.pdf")
    
    # Print summary
    print("\n📋 QUICK SUMMARY:")
    print(f"Total Benchmarks: {len(df)}")
    print(f"Languages: {', '.join(df['language'].unique())}")
    print(f"Task Types: {', '.join(df['task_type'].unique())}")
    print(f"Algorithms: {len(df['algorithm'].unique())}")
    
    # Performance summary
    python_avg_time = df[df['language']=='python']['training_time_seconds'].mean()
    rust_avg_time = df[df['language']=='rust']['training_time_seconds'].mean()
    python_avg_memory = df[df['language']=='python']['peak_memory_mb'].mean()
    rust_avg_memory = df[df['language']=='rust']['peak_memory_mb'].mean()
    
    print(f"\n⚡ Performance Comparison:")
    print(f"Python Avg Training Time: {python_avg_time:.4f}s")
    print(f"Rust Avg Training Time: {rust_avg_time:.4f}s")
    print(f"Rust Improvement: {((python_avg_time - rust_avg_time) / python_avg_time * 100):.1f}%")
    print(f"\n💾 Memory Comparison:")
    print(f"Python Avg Memory: {python_avg_memory:.1f} MB")
    print(f"Rust Avg Memory: {rust_avg_memory:.1f} MB")
    print(f"Rust Improvement: {((python_avg_memory - rust_avg_memory) / python_avg_memory * 100):.1f}%")

if __name__ == "__main__":
    main() 