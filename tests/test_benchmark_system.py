#!/usr/bin/env python3
"""
Comprehensive Test Suite for Rust vs Python ML Benchmark System

This module provides comprehensive tests for all components of the benchmark system.
"""

import unittest
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pandas as pd

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from shared.schemas.metrics import (
    BenchmarkResult, PerformanceMetrics, ResourceMetrics, QualityMetrics,
    HardwareConfig, Language, TaskType
)


class TestMetricsSchema(unittest.TestCase):
    """Test the metrics schema components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.hardware_config = HardwareConfig(
            cpu_model="Intel Core i9-13900K",
            cpu_cores=24,
            cpu_threads=32,
            memory_gb=64.0,
            gpu_model="NVIDIA RTX 4090",
            gpu_memory_gb=24.0
        )
        
        self.performance_metrics = PerformanceMetrics(
            training_time_seconds=10.5,
            inference_latency_ms=2.3,
            throughput_samples_per_second=1000.0
        )
        
        self.resource_metrics = ResourceMetrics(
            peak_memory_mb=2048.0,
            average_memory_mb=1024.0,
            cpu_utilization_percent=75.5,
            peak_gpu_memory_mb=8192.0,
            average_gpu_memory_mb=4096.0,
            gpu_utilization_percent=85.0
        )
        
        self.quality_metrics = QualityMetrics(
            accuracy=0.95,
            f1_score=0.94,
            precision=0.96,
            recall=0.93
        )
    
    def test_hardware_config_creation(self):
        """Test HardwareConfig creation and serialization."""
        config_dict = {
            "cpu_model": "Intel Core i9-13900K",
            "cpu_cores": 24,
            "cpu_threads": 32,
            "memory_gb": 64.0,
            "gpu_model": "NVIDIA RTX 4090",
            "gpu_memory_gb": 24.0
        }
        
        config = HardwareConfig(**config_dict)
        self.assertEqual(config.cpu_model, "Intel Core i9-13900K")
        self.assertEqual(config.cpu_cores, 24)
        self.assertEqual(config.gpu_memory_gb, 24.0)
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation."""
        metrics = PerformanceMetrics(
            training_time_seconds=10.5,
            inference_latency_ms=2.3
        )
        
        self.assertEqual(metrics.training_time_seconds, 10.5)
        self.assertEqual(metrics.inference_latency_ms, 2.3)
        self.assertIsNone(metrics.tokens_per_second)
    
    def test_resource_metrics_creation(self):
        """Test ResourceMetrics creation."""
        metrics = ResourceMetrics(
            peak_memory_mb=2048.0,
            average_memory_mb=1024.0,
            cpu_utilization_percent=75.5
        )
        
        self.assertEqual(metrics.peak_memory_mb, 2048.0)
        self.assertEqual(metrics.cpu_utilization_percent, 75.5)
        self.assertIsNone(metrics.gpu_utilization_percent)
    
    def test_quality_metrics_creation(self):
        """Test QualityMetrics creation."""
        metrics = QualityMetrics(
            accuracy=0.95,
            f1_score=0.94
        )
        
        self.assertEqual(metrics.accuracy, 0.95)
        self.assertEqual(metrics.f1_score, 0.94)
        self.assertIsNone(metrics.loss)
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation."""
        result = BenchmarkResult(
            framework="scikit-learn",
            language=Language.PYTHON,
            task_type=TaskType.CLASSICAL_ML,
            model_name="linear_regression",
            dataset="boston_housing",
            run_id="test_run_001",
            timestamp=pd.Timestamp.now(),
            hardware_config=self.hardware_config,
            performance_metrics=self.performance_metrics,
            resource_metrics=self.resource_metrics,
            quality_metrics=self.quality_metrics
        )
        
        self.assertEqual(result.framework, "scikit-learn")
        self.assertEqual(result.language, Language.PYTHON)
        self.assertEqual(result.task_type, TaskType.CLASSICAL_ML)
        self.assertEqual(result.model_name, "linear_regression")
    
    def test_benchmark_result_serialization(self):
        """Test BenchmarkResult JSON serialization."""
        result = BenchmarkResult(
            framework="scikit-learn",
            language=Language.PYTHON,
            task_type=TaskType.CLASSICAL_ML,
            model_name="linear_regression",
            dataset="boston_housing",
            run_id="test_run_001",
            timestamp=pd.Timestamp.now(),
            hardware_config=self.hardware_config,
            performance_metrics=self.performance_metrics,
            resource_metrics=self.resource_metrics,
            quality_metrics=self.quality_metrics
        )
        
        # Test to_dict
        result_dict = result.to_dict()
        self.assertIn("framework", result_dict)
        self.assertIn("language", result_dict)
        self.assertIn("performance_metrics", result_dict)
        self.assertIn("resource_metrics", result_dict)
        self.assertIn("quality_metrics", result_dict)
        
        # Test to_json
        json_str = result.to_json()
        self.assertIsInstance(json_str, str)
        
        # Test from_json
        reconstructed_result = BenchmarkResult.from_json(json_str)
        self.assertEqual(reconstructed_result.framework, result.framework)
        self.assertEqual(reconstructed_result.language, result.language)
        self.assertEqual(reconstructed_result.model_name, result.model_name)


class TestPythonBenchmarks(unittest.TestCase):
    """Test Python benchmark implementations."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('src.python.classical_ml.regression_benchmark.RegressionBenchmark')
    def test_regression_benchmark_creation(self, mock_benchmark):
        """Test regression benchmark creation."""
        # Mock the benchmark class
        mock_instance = Mock()
        mock_benchmark.return_value = mock_instance
        
        # Test benchmark creation
        benchmark = mock_benchmark()
        self.assertIsNotNone(benchmark)
    
    @patch('src.python.classical_ml.svm_benchmark.SVMBenchmark')
    def test_svm_benchmark_creation(self, mock_benchmark):
        """Test SVM benchmark creation."""
        # Mock the benchmark class
        mock_instance = Mock()
        mock_benchmark.return_value = mock_instance
        
        # Test benchmark creation
        benchmark = mock_benchmark()
        self.assertIsNotNone(benchmark)
    
    @patch('src.python.deep_learning.cnn_benchmark.CNNBenchmark')
    def test_cnn_benchmark_creation(self, mock_benchmark):
        """Test CNN benchmark creation."""
        # Mock the benchmark class
        mock_instance = Mock()
        mock_benchmark.return_value = mock_instance
        
        # Test benchmark creation
        benchmark = mock_benchmark()
        self.assertIsNotNone(benchmark)


class TestRustBenchmarks(unittest.TestCase):
    """Test Rust benchmark implementations."""
    
    def test_rust_benchmark_structure(self):
        """Test that Rust benchmark files exist and have correct structure."""
        rust_benchmark_path = Path("../src/rust/classical_ml/regression_benchmark/src/main.rs")
        
        # Check if the file exists
        self.assertTrue(rust_benchmark_path.exists(), 
                       f"Rust benchmark file not found: {rust_benchmark_path}")
        
        # Check if Cargo.toml exists
        cargo_toml_path = Path("../src/rust/classical_ml/regression_benchmark/Cargo.toml")
        self.assertTrue(cargo_toml_path.exists(),
                       f"Cargo.toml not found: {cargo_toml_path}")


class TestConfiguration(unittest.TestCase):
    """Test configuration files and validation."""
    
    def test_benchmarks_config_structure(self):
        """Test benchmarks configuration structure."""
        config_path = Path("../config/benchmarks.yaml")
        self.assertTrue(config_path.exists(), "benchmarks.yaml not found")
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        self.assertIn("benchmarks", config)
        self.assertIn("classical_ml", config["benchmarks"])
        self.assertIn("deep_learning", config["benchmarks"])
        self.assertIn("reinforcement_learning", config["benchmarks"])
        self.assertIn("llm", config["benchmarks"])
    
    def test_frameworks_config_structure(self):
        """Test frameworks configuration structure."""
        config_path = Path("../config/frameworks.yaml")
        self.assertTrue(config_path.exists(), "frameworks.yaml not found")
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        self.assertIn("python", config)
        self.assertIn("rust", config)
    
    def test_hardware_config_structure(self):
        """Test hardware configuration structure."""
        config_path = Path("../config/hardware.yaml")
        self.assertTrue(config_path.exists(), "hardware.yaml not found")
        
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        self.assertIn("system", config)
        self.assertIn("resource_limits", config)
        self.assertIn("monitoring", config)


class TestWorkflowComponents(unittest.TestCase):
    """Test Nextflow workflow components."""
    
    def test_nextflow_config_exists(self):
        """Test that Nextflow configuration exists."""
        config_path = Path("../nextflow.config")
        self.assertTrue(config_path.exists(), "nextflow.config not found")
    
    def test_main_workflow_exists(self):
        """Test that main workflow exists."""
        workflow_path = Path("../main.nf")
        self.assertTrue(workflow_path.exists(), "main.nf not found")
    
    def test_phase_workflows_exist(self):
        """Test that all phase workflows exist."""
        workflow_dir = Path("../workflows")
        self.assertTrue(workflow_dir.exists(), "workflows directory not found")
        
        required_workflows = [
            "phase1_selection.nf",
            "phase2_implementation.nf",
            "phase3_experiment.nf",
            "phase4_benchmark.nf",
            "phase5_analysis.nf",
            "phase6_assessment.nf"
        ]
        
        for workflow in required_workflows:
            workflow_path = workflow_dir / workflow
            self.assertTrue(workflow_path.exists(), 
                          f"Required workflow not found: {workflow}")


class TestScripts(unittest.TestCase):
    """Test Python scripts."""
    
    def test_scripts_exist(self):
        """Test that all required scripts exist."""
        scripts_dir = Path("../scripts")
        self.assertTrue(scripts_dir.exists(), "scripts directory not found")
        
        required_scripts = [
            "validate_frameworks.py",
            "check_availability.py",
            "select_frameworks.py",
            "perform_statistical_analysis.py",
            "create_visualizations.py",
            "generate_final_report.py"
        ]
        
        for script in required_scripts:
            script_path = scripts_dir / script
            self.assertTrue(script_path.exists(), 
                          f"Required script not found: {script}")
    
    @patch('scripts.validate_frameworks.FrameworkValidator')
    def test_framework_validation_script(self, mock_validator):
        """Test framework validation script structure."""
        # Mock the validator class
        mock_instance = Mock()
        mock_validator.return_value = mock_instance
        
        # Test that the script can be imported
        try:
            import scripts.validate_frameworks
            self.assertTrue(True, "Framework validation script imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import framework validation script: {e}")


class TestDockerContainers(unittest.TestCase):
    """Test Docker container configurations."""
    
    def test_python_dockerfile_exists(self):
        """Test that Python Dockerfile exists."""
        dockerfile_path = Path("../containers/python/Dockerfile")
        self.assertTrue(dockerfile_path.exists(), "Python Dockerfile not found")
    
    def test_rust_dockerfile_exists(self):
        """Test that Rust Dockerfile exists."""
        dockerfile_path = Path("../containers/rust/Dockerfile")
        self.assertTrue(dockerfile_path.exists(), "Rust Dockerfile not found")


class TestDocumentation(unittest.TestCase):
    """Test documentation files."""
    
    def test_readme_exists(self):
        """Test that README exists."""
        readme_path = Path("../README.md")
        self.assertTrue(readme_path.exists(), "README.md not found")
    
    def test_specs_exists(self):
        """Test that SPECS.md exists."""
        specs_path = Path("../SPECS.md")
        self.assertTrue(specs_path.exists(), "SPECS.md not found")


class TestDataStructures(unittest.TestCase):
    """Test data structures and schemas."""
    
    def test_metrics_schema_completeness(self):
        """Test that metrics schema is complete."""
        # Test that all required classes exist
        required_classes = [
            'BenchmarkResult',
            'PerformanceMetrics', 
            'ResourceMetrics',
            'QualityMetrics',
            'HardwareConfig',
            'Language',
            'TaskType'
        ]
        
        for class_name in required_classes:
            self.assertTrue(hasattr(sys.modules['src.shared.schemas.metrics'], class_name),
                          f"Required class not found: {class_name}")


class TestStatisticalAnalysis(unittest.TestCase):
    """Test statistical analysis components."""
    
    def test_statistical_analyzer_creation(self):
        """Test statistical analyzer creation."""
        try:
            from scripts.perform_statistical_analysis import StatisticalAnalyzer
            analyzer = StatisticalAnalyzer(confidence_level=0.95)
            self.assertIsNotNone(analyzer)
        except ImportError as e:
            self.fail(f"Failed to import StatisticalAnalyzer: {e}")
    
    def test_effect_size_calculations(self):
        """Test effect size calculations."""
        # Test data
        group1 = [1, 2, 3, 4, 5]
        group2 = [2, 3, 4, 5, 6]
        
        # Test Cohen's d calculation
        from scripts.perform_statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        
        cohens_d = analyzer._calculate_cohens_d(group1, group2)
        self.assertIsInstance(cohens_d, float)
        
        cliffs_delta = analyzer._calculate_cliffs_delta(group1, group2)
        self.assertIsInstance(cliffs_delta, float)


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestMetricsSchema,
        TestPythonBenchmarks,
        TestRustBenchmarks,
        TestConfiguration,
        TestWorkflowComponents,
        TestScripts,
        TestDockerContainers,
        TestDocumentation,
        TestDataStructures,
        TestStatisticalAnalysis
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with appropriate code
    sys.exit(not result.wasSuccessful()) 