import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class HealthAnomalyDetector:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = ['heart_rate', 'blood_oxygen', 'steps']
        
    def prepare_features(self, df):
        """Prepare features for anomaly detection"""
        
        # Create additional features
        df_features = df.copy()
        
        # Time-based features
        df_features['hour'] = pd.to_datetime(df_features['timestamp']).dt.hour
        df_features['day_of_week'] = pd.to_datetime(df_features['timestamp']).dt.dayofweek
        
        # Rolling averages (24 hours = 288 data points at 5-min intervals)
        df_features['hr_rolling_mean'] = df_features['heart_rate'].rolling(window=12, min_periods=1).mean()
        df_features['hr_rolling_std'] = df_features['heart_rate'].rolling(window=12, min_periods=1).std()
        
        # Heart rate variability
        df_features['hr_change'] = df_features['heart_rate'].diff().abs()
        
        # Activity encoding
        activity_encoding = {'low': 0, 'moderate': 1, 'high': 2}
        df_features['activity_encoded'] = df_features['activity_level'].map(activity_encoding)
        
        # Select features for model
        feature_cols = ['heart_rate', 'blood_oxygen', 'steps', 'hour', 
                       'hr_rolling_mean', 'hr_change', 'activity_encoded']
        
        return df_features[feature_cols].fillna(method='ffill').fillna(0)
    
    def train_model(self, df, contamination=0.1):
        """Train the anomaly detection model"""
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        # Fit the model
        predictions = self.model.fit_predict(X_scaled)
        
        # Add predictions to original dataframe
        df_result = df.copy()
        df_result['anomaly_score'] = self.model.score_samples(X_scaled)
        df_result['is_anomaly'] = (predictions == -1)
        
        return df_result
    
    def predict(self, df):
        """Predict anomalies on new data"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        df_result = df.copy()
        df_result['anomaly_score'] = scores
        df_result['is_anomaly'] = (predictions == -1)
        
        return df_result
    
    def save_model(self, filepath):
        """Save trained model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler
        }
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath):
        """Load trained model and scaler"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
    
    def plot_anomalies(self, df_with_predictions):
        """Visualize anomalies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Heart rate over time
        axes[0, 0].plot(df_with_predictions['timestamp'], 
                       df_with_predictions['heart_rate'], 
                       alpha=0.7, label='Normal')
        anomalies = df_with_predictions[df_with_predictions['is_anomaly']]
        axes[0, 0].scatter(anomalies['timestamp'], 
                          anomalies['heart_rate'], 
                          color='red', label='Anomaly', s=50)
        axes[0, 0].set_title('Heart Rate Anomalies')
        axes[0, 0].legend()
        
        # Blood oxygen over time
        axes[0, 1].plot(df_with_predictions['timestamp'], 
                       df_with_predictions['blood_oxygen'], 
                       alpha=0.7, label='Normal')
        axes[0, 1].scatter(anomalies['timestamp'], 
                          anomalies['blood_oxygen'], 
                          color='red', label='Anomaly', s=50)
        axes[0, 1].set_title('Blood Oxygen Anomalies')
        axes[0, 1].legend()
        
        # Anomaly score distribution
        axes[1, 0].hist(df_with_predictions['anomaly_score'], bins=50, alpha=0.7)
        axes[1, 0].axvline(df_with_predictions[df_with_predictions['is_anomaly']]['anomaly_score'].max(), 
                          color='red', linestyle='--', label='Anomaly Threshold')
        axes[1, 0].set_title('Anomaly Score Distribution')
        axes[1, 0].legend()
        
        # Scatter plot
        scatter = axes[1, 1].scatter(df_with_predictions['heart_rate'], 
                                   df_with_predictions['blood_oxygen'],
                                   c=df_with_predictions['anomaly_score'], 
                                   cmap='viridis', alpha=0.6)
        axes[1, 1].scatter(anomalies['heart_rate'], 
                          anomalies['blood_oxygen'],
                          color='red', s=100, alpha=0.8)
        axes[1, 1].set_xlabel('Heart Rate')
        axes[1, 1].set_ylabel('Blood Oxygen')
        axes[1, 1].set_title('Heart Rate vs Blood Oxygen')
        plt.colorbar(scatter, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('static/anomaly_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

# Training script
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('data/health_data.csv')
    
    # Initialize detector
    detector = HealthAnomalyDetector()
    
    # Train model
    df_with_predictions = detector.train_model(df)
    
    # Save model
    detector.save_model('models/anomaly_model.pkl')
    
    # Print results
    anomaly_count = df_with_predictions['is_anomaly'].sum()
    total_count = len(df_with_predictions)
    print(f"Detected {anomaly_count} anomalies out of {total_count} records ({anomaly_count/total_count*100:.2f}%)")
    
    # Show some anomalies
    print("\nSample anomalies:")
    print(df_with_predictions[df_with_predictions['is_anomaly']][['timestamp', 'heart_rate', 'blood_oxygen', 'anomaly_score']].head())
    
    # Plot results
    detector.plot_anomalies(df_with_predictions)
