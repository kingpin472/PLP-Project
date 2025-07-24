import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class HealthRecommendationEngine:
    def __init__(self):
        self.recommendations_db = {
            'high_heart_rate': [
                "Consider taking a break and practicing deep breathing exercises.",
                "If this persists, consult with your healthcare provider.",
                "Avoid caffeine and strenuous activities temporarily."
            ],
            'low_heart_rate': [
                "Monitor your symptoms. If you feel dizzy or weak, seek medical attention.",
                "Ensure you're getting adequate sleep and nutrition.",
                "Consider light physical activity if you feel well."
            ],
            'low_blood_oxygen': [
                "Check if you're in a well-ventilated area.",
                "Practice deep breathing exercises.",
                "If symptoms persist, consult a healthcare professional immediately."
            ],
            'low_activity': [
                "Try to incorporate more movement into your day.",
                "Consider taking short walks every hour.",
                "Set activity reminders on your device."
            ],
            'good_health': [
                "Great job maintaining your health metrics!",
                "Keep up your current activity level.",
                "Remember to stay hydrated throughout the day."
            ]
        }
    
    def analyze_health_status(self, recent_data):
        """Analyze recent health data and provide recommendations"""
        
        if len(recent_data) == 0:
            return {"status": "insufficient_data", "recommendations": ["Please ensure your device is collecting data."]}
        
        # Get latest values
        latest_hr = recent_data['heart_rate'].iloc[-1]
        latest_oxygen = recent_data['blood_oxygen'].iloc[-1]
        avg_steps = recent_data['steps'].mean()
        
        # Check for anomalies in recent data
        recent_anomalies = recent_data['is_anomaly'].sum() if 'is_anomaly' in recent_data.columns else 0
        
        recommendations = []
        alerts = []
        status = "normal"
        
        # Heart rate analysis
        if latest_hr > 100:
            status = "attention_needed"
            recommendations.extend(self.recommendations_db['high_heart_rate'])
            alerts.append(f"High heart rate detected: {latest_hr} bpm")
        elif latest_hr < 50:
            status = "attention_needed"
            recommendations.extend(self.recommendations_db['low_heart_rate'])
            alerts.append(f"Low heart rate detected: {latest_hr} bpm")
        
        # Blood oxygen analysis
        if latest_oxygen < 95:
            status = "attention_needed"
            recommendations.extend(self.recommendations_db['low_blood_oxygen'])
            alerts.append(f"Low blood oxygen: {latest_oxygen}%")
        
        # Activity analysis
        if avg_steps < 2000:  # Low activity
            recommendations.extend(self.recommendations_db['low_activity'])
        
        # Anomaly analysis
        if recent_anomalies > 0:
            status = "attention_needed"
            alerts.append(f"{recent_anomalies} health anomalies detected in recent data")
        
        # If everything looks good
        if status == "normal":
            recommendations.extend(self.recommendations_db['good_health'])
        
        # Remove duplicates
        recommendations = list(set(recommendations))
        
        return {
            "status": status,
            "latest_heart_rate": latest_hr,
            "latest_blood_oxygen": latest_oxygen,
            "average_daily_steps": int(avg_steps),
            "recent_anomalies": recent_anomalies,
            "recommendations": recommendations,
            "alerts": alerts
        }
    
    def generate_weekly_report(self, weekly_data):
        """Generate a comprehensive weekly health report"""
        
        if len(weekly_data) == 0:
            return {"error": "No data available for weekly report"}
        
        # Calculate statistics
        stats = {
            "avg_heart_rate": weekly_data['heart_rate'].mean(),
            "min_heart_rate": weekly_data['heart_rate'].min(),
            "max_heart_rate": weekly_data['heart_rate'].max(),
            "avg_blood_oxygen": weekly_data['blood_oxygen'].mean(),
            "min_blood_oxygen": weekly_data['blood_oxygen'].min(),
            "total_steps": weekly_data['steps'].sum(),
            "avg_daily_steps": weekly_data['steps'].sum() / 7,
            "total_anomalies": weekly_data['is_anomaly'].sum() if 'is_anomaly' in weekly_data.columns else 0
        }
        
        # Health trends
        trends = self._analyze_trends(weekly_data)
        
        # Generate insights
        insights = self._generate_insights(stats, trends)
        
        return {
            "period": "Past 7 days",
            "statistics": stats,
            "trends": trends,
            "insights": insights,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _analyze_trends(self, data):
        """Analyze health trends over time"""
        
        # Group by day
        daily_data = data.groupby(data['timestamp'].dt.date).agg({
            'heart_rate': 'mean',
            'blood_oxygen': 'mean',
            'steps': 'sum'
        }).reset_index()
        
        trends = {}
        
        if len(daily_data) >= 2:
            # Calculate trends (positive = improving, negative = declining)
            hr_trend = (daily_data['heart_rate'].iloc[-1] - daily_data['heart_rate'].iloc[0]) / len(daily_data)
            oxygen_trend = (daily_data['blood_oxygen'].iloc[-1] - daily_data['blood_oxygen'].iloc[0]) / len(daily_data)
            steps_trend = (daily_data['steps'].iloc[-1] - daily_data['steps'].iloc[0]) / len(daily_data)
            
            trends = {
                "heart_rate": "increasing" if hr_trend > 1 else "decreasing" if hr_trend < -1 else "stable",
                "blood_oxygen": "increasing" if oxygen_trend > 0.5 else "decreasing" if oxygen_trend < -0.5 else "stable",
                "activity": "increasing" if steps_trend > 200 else "decreasing" if steps_trend < -200 else "stable"
            }
        
        return trends
    
    def _generate_insights(self, stats, trends):
        """Generate health insights based on statistics and trends"""
        
        insights = []
        
        # Heart rate insights
        if stats['avg_heart_rate'] >= 60 and stats['avg_heart_rate'] <= 100:
            insights.append("Your average heart rate is within the normal range.")
        else:
            insights.append("Your average heart rate is outside the typical range. Consider consulting a healthcare provider.")
        
        # Blood oxygen insights
        if stats['avg_blood_oxygen'] >= 95:
            insights.append("Your blood oxygen levels are healthy.")
        else:
            insights.append("Your blood oxygen levels are below optimal. This may need medical attention.")
        
        # Activity insights
        daily_steps = stats['avg_daily_steps']
        if daily_steps >= 10000:
            insights.append("Excellent! You're meeting the recommended daily step goal.")
        elif daily_steps >= 7500:
            insights.append("Good activity level! Try to reach 10,000 steps daily.")
        else:
            insights.append("Consider increasing your daily physical activity.")
        
        # Trend insights
        if trends.get('heart_rate') == 'increasing':
            insights.append("Your heart rate has been trending upward - monitor for any symptoms.")
        
        if trends.get('activity') == 'decreasing':
            insights.append("Your activity level has been decreasing - try to stay more active.")
        
        return insights
