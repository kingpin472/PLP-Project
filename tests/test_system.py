import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_generator import HealthDataGenerator
from models.anomaly_detector import HealthAnomalyDetector
from models.recommendation_engine import HealthRecommendationEngine

class TestHealthMonitoringSystem(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.generator = HealthDataGenerator()
        self.detector = HealthAnomalyDetector()
        self.recommender = HealthRecommendationEngine()
        
        # Generate test data
        self.test_data = self.generator.generate_realistic_data(days=7)
    
    def test_data_generation(self):
        """Test that data generation works correctly."""
        data = self.generator.generate_realistic_data(days=1)
        
        self.assertIsInstance(data, pd.DataFrame)
        self.assertTrue(len(data) > 0)
        self.assertTrue(all(col in data.columns for col in 
                          ['timestamp', 'heart_rate', 'blood_oxygen', 'steps']))
        
        # Check data ranges
        self.assertTrue(data['heart_rate'].between(30, 200).all())
        self.assertTrue(data['blood_oxygen'].between(80, 100).all())
        self.assertTrue(data['steps'].min() >= 0)
    
    def test_anomaly_detection(self):
        """Test anomaly detection model."""
        # Train model
        result_data = self.detector.train_model(self.test_data)
        
        self.assertIsInstance(result_data, pd.DataFrame)
        self.assertTrue('is_anomaly' in result_data.columns)
        self.assertTrue('anomaly_score' in result_data.columns)
        
        # Check that we detect some anomalies (but not too many)
        anomaly_rate = result_data['is_anomaly'].mean()
        self.assertTrue(0.01 <= anomaly_rate <= 0.2)  # Between 1% and 20%
    
    def test_recommendations(self):
        """Test recommendation engine."""
        # Add anomaly predictions to test data
        test_data_with_anomalies = self.detector.train_model(self.test_data)
        
        # Test current status analysis
        recent_data = test_data_with_anomalies.tail(50)  # Last 50 records
        analysis = self.recommender.analyze_health_status(recent_data)
        
        self.assertIn('status', analysis)
        self.assertIn('recommendations', analysis)
        self.assertIn('latest_heart_rate', analysis)
        self.assertIn('latest_blood_oxygen', analysis)
        
        self.assertIsInstance(analysis['recommendations'], list)
    
    def test_weekly_report(self):
        """Test weekly report generation."""
        test_data_with_anomalies = self.detector.train_model(self.test_data)
        
        report = self.recommender.generate_weekly_report(test_data_with_anomalies)
        
        self.assertIn('statistics', report)
        self.assertIn('insights', report)
        self.assertIn('trends', report)
        
        stats = report['statistics']
        self.assertTrue(stats['avg_heart_rate'] > 0)
        self.assertTrue(0 <= stats['avg_blood_oxygen'] <= 100)
        self.assertTrue(stats['total_steps'] >= 0)
    
    def test_model_persistence(self):
        """Test that model can be saved and loaded."""
        # Train and save model
        self.detector.train_model(self.test_data)
        self.detector.save_model('test_model.pkl')
        
        # Create new detector and load model
        new_detector = HealthAnomalyDetector()
        new_detector.load_model('test_model.pkl')
        
        # Test prediction on new data
        new_data = self.generator.generate_realistic_data(days=1)
        predictions = new_detector.predict(new_data)
        
        self.assertTrue('is_anomaly' in predictions.columns)
        
        # Clean up
        os.remove('test_model.pkl')
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Empty dataframe
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        analysis = self.recommender.analyze_health_status(empty_df)
        self.assertEqual(analysis['status'], 'insufficient_data')
        
        # Single data point
        single_point = self.test_data.iloc[:1].copy()
        analysis = self.recommender.analyze_health_status(single_point)
        self.assertIn('status', analysis)

if __name__ == '__main__':
    # Create test directory if it doesn't exist
    os.makedirs('tests', exist_ok=True)
    
    # Run tests
    unittest.main(verbosity=2)
