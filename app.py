from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
from datetime import datetime, timedelta
from models.anomaly_detector import HealthAnomalyDetector
from models.recommendation_engine import HealthRecommendationEngine
import plotly
import plotly.graph_objs as go
import plotly.express as px

app = Flask(__name__)

# Initialize models
detector = HealthAnomalyDetector()
recommender = HealthRecommendationEngine()

# Load trained model
try:
    detector.load_model('models/anomaly_model.pkl')
    print("Anomaly detection model loaded successfully")
except:
    print("Warning: Could not load anomaly detection model. Please train the model first.")

# Load sample data
def load_health_data():
    try:
        df = pd.read_csv('data/health_data.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except:
        return pd.DataFrame()

@app.route('/')
def dashboard():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/current_status')
def current_status():
    """Get current health status and recommendations"""
    
    df = load_health_data()
    if df.empty:
        return jsonify({"error": "No data available"})
    
    # Get recent data (last 24 hours)
    recent_cutoff = df['timestamp'].max() - timedelta(hours=24)
    recent_data = df[df['timestamp'] >= recent_cutoff].copy()
    
    # Run anomaly detection on recent data
    if detector.model is not None:
        recent_data = detector.predict(recent_data)
    
    # Get recommendations
    analysis = recommender.analyze_health_status(recent_data)
    
    return jsonify(analysis)

@app.route('/api/health_chart')
def health_chart():
    """Generate health data chart"""
    
    df = load_health_data()
    if df.empty:
        return jsonify({"error": "No data available"})
    
    # Get last 7 days of data
    recent_cutoff = df['timestamp'].max() - timedelta(days=7)
    chart_data = df[df['timestamp'] >= recent_cutoff].copy()
    
    # Run anomaly detection
    if detector.model is not None:
        chart_data = detector.predict(chart_data)
    
    # Create heart rate chart
    fig = go.Figure()
    
    # Normal data points
    normal_data = chart_data[~chart_data.get('is_anomaly', False)]
    fig.add_trace(go.Scatter(
        x=normal_data['timestamp'],
        y=normal_data['heart_rate'],
        mode='lines+markers',
        name='Heart Rate',
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    # Anomaly data points
    if 'is_anomaly' in chart_data.columns:
        anomaly_data = chart_data[chart_data['is_anomaly']]
        if not anomaly_data.empty:
            fig.add_trace(go.Scatter(
                x=anomaly_data['timestamp'],
                y=anomaly_data['heart_rate'],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='x')
            ))
    
    fig.update_layout(
        title='Heart Rate - Last 7 Days',
        xaxis_title='Time',
        yaxis_title='Heart Rate (BPM)',
        hovermode='x unified'
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return jsonify({"chart": graphJSON})

@app.route('/api/weekly_report')
def weekly_report():
    """Generate weekly health report"""
    
    df = load_health_data()
    if df.empty:
        return jsonify({"error": "No data available"})
    
    # Get last 7 days of data
    week_cutoff = df['timestamp'].max() - timedelta(days=7)
    weekly_data = df[df['timestamp'] >= week_cutoff].copy()
    
    # Run anomaly detection
    if detector.model is not None:
        weekly_data = detector.predict(weekly_data)
    
    # Generate report
    report = recommender.generate_weekly_report(weekly_data)
    
    return jsonify(report)

@app.route('/api/simulate_realtime')
def simulate_realtime():
    """Simulate real-time data update"""
    
    from data.data_generator import HealthDataGenerator
    
    generator = HealthDataGenerator()
    
    # Generate one new data point
    new_data = generator.generate_realistic_data(days=1, frequency_minutes=5)
    latest_point = new_data.iloc[-1]
    
    # Run anomaly detection on the latest point
    if detector.model is not None:
        prediction = detector.predict(pd.DataFrame([latest_point]))
        latest_point = prediction.iloc[0]
    
    return jsonify({
        "timestamp": latest_point['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
        "heart_rate": int(latest_point['heart_rate']),
        "blood_oxygen": float(latest_point['blood_oxygen']),
        "steps": int(latest_point['steps']),
        "activity_level": latest_point['activity_level'],
        "is_anomaly": latest_point.get('is_anomaly', False),
        "anomaly_score": float(latest_point.get('anomaly_score', 0))
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
