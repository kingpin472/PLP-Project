import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class HealthDataGenerator:
    def __init__(self):
        self.base_heart_rate = 70
        self.base_blood_oxygen = 98
        
    def generate_realistic_data(self, days=30, frequency_minutes=5):
        """Generate realistic health data with patterns and anomalies"""
        
        # Calculate total data points
        total_minutes = days * 24 * 60
        data_points = total_minutes // frequency_minutes
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(days=days)
        timestamps = [start_time + timedelta(minutes=i*frequency_minutes) 
                     for i in range(data_points)]
        
        data = []
        
        for i, timestamp in enumerate(timestamps):
            # Time-based patterns
            hour = timestamp.hour
            
            # Heart rate patterns (lower at night, higher during day)
            if 22 <= hour or hour <= 6:  # Night time
                hr_base = self.base_heart_rate - 10
                activity = 'low'
            elif 6 < hour < 9 or 17 < hour < 20:  # Active periods
                hr_base = self.base_heart_rate + 15
                activity = 'high'
            else:  # Regular day
                hr_base = self.base_heart_rate
                activity = 'moderate'
            
            # Add natural variation
            heart_rate = max(45, min(120, 
                np.random.normal(hr_base, 8)))
            
            # Blood oxygen (more stable, occasional drops)
            blood_oxygen = np.random.normal(self.base_blood_oxygen, 1.5)
            blood_oxygen = max(85, min(100, blood_oxygen))
            
            # Introduce some anomalies (5% chance)
            if random.random() < 0.05:
                if random.random() < 0.5:
                    heart_rate = random.choice([
                        random.randint(35, 50),  # Bradycardia
                        random.randint(110, 140)  # Tachycardia
                    ])
                else:
                    blood_oxygen = random.randint(85, 92)  # Low oxygen
            
            # Steps (correlate with activity)
            if activity == 'high':
                steps = random.randint(500, 1500)
            elif activity == 'moderate':
                steps = random.randint(100, 500)
            else:
                steps = random.randint(0, 100)
            
            data.append({
                'timestamp': timestamp,
                'heart_rate': round(heart_rate),
                'blood_oxygen': round(blood_oxygen, 1),
                'activity_level': activity,
                'steps': steps,
                'user_id': 1  # Single user for now
            })
        
        return pd.DataFrame(data)

# Generate and save sample data
if __name__ == "__main__":
    generator = HealthDataGenerator()
    df = generator.generate_realistic_data(days=30)
    df.to_csv('data/health_data.csv', index=False)
    print(f"Generated {len(df)} health records")
    print(df.head())
