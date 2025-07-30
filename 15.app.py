# app.py - Enhanced Maritime Anomaly Detection API with 70/30 Hybrid Scoring
import os
import time
import traceback
import logging
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from shapely.geometry import Point, Polygon
from haversine import haversine, Unit
from sklearn.preprocessing import MinMaxScaler

# --- Bootstrap ---
app = Flask(__name__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - API - %(levelname)s - %(message)s')

# --- Hybrid Scoring Configuration ---
ML_WEIGHT = 0.7      # 70% ML contribution
RULE_WEIGHT = 0.3    # 30% Rule-based contribution
HYBRID_THRESHOLD = 0.30  # Anomaly threshold for hybrid score

# --- Model Configuration ---
MODEL_PATH_DIR = '/Users/hafizmohammedaahil/Documents/C3I-Code/C3I-Internship-Work/Programs/MaritimeAnomalyDetection/models/latest-model/'
SEQ_LEN = 10
FEATURES = ['latitude', 'longitude', 'speed', 'course', 'heading']

# --- Define Geofenced Zones ---
NORWEGIAN_EEZ = Polygon([(-0.4907, 56.086), (36.4763, 56.086), (36.4763, 74.5048), (-0.4907, 74.5048)])
SWEDISH_EEZ_BALTIC = Polygon([(10.03, 54.9624), (24.1897, 54.9624), (24.1897, 67.0806), (10.03, 67.0806)])
MARINE_PROTECTED_AREA = Polygon([(14.0, 68.5), (15.5, 68.5), (15.5, 69.0), (14.0, 69.0)])
PORT_OF_ROGNAN = Polygon([(15.35, 67.08), (15.45, 67.08), (15.45, 67.12), (15.35, 67.12)])
SAFE_ZONES = [PORT_OF_ROGNAN]

# --- Load ML Models ---
try:
    logging.info("Loading Enhanced-GRU-AE models...")
    encoder = tf.keras.models.load_model(os.path.join(MODEL_PATH_DIR, 'gru_encoder.keras'))
    detector = joblib.load(os.path.join(MODEL_PATH_DIR, 'final_anomaly_detector.pkl'))
    scaler = joblib.load(os.path.join(MODEL_PATH_DIR, 'final_data_scaler.pkl'))
    logging.info("✅ All models loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Could not load models. Error: {e}")
    exit(1)

# --- Scoring Normalization ---
score_normalizer = MinMaxScaler((0, 1)).fit(np.logspace(-4, 2, 50).reshape(-1, 1))

def normalize_ml_score(ml_decision):
    """Convert SVM decision function output to [0,1] score"""
    # SVM outputs negative for normal, positive for anomaly
    # Convert to 0-1 scale where higher = more anomalous
    normalized = max(0, min(1, (0.5 - ml_decision) / 2))
    return normalized

def calculate_rule_score(sequence_df):
    """
    Calculate rule-based anomaly score and return violations
    Returns: (normalized_score, list_of_rule_violations)
    """
    last_point = sequence_df.iloc[-1]
    current_position = Point(last_point['longitude'], last_point['latitude'])
    
    # Skip scoring if in safe zones
    for safe_zone in SAFE_ZONES:
        if safe_zone.contains(current_position):
            return 0.0, []
    
    rule_score = 0.0
    violations = []
    
    # R1: Impossible position jump detection
    if len(sequence_df) > 1:
        prev_point = sequence_df.iloc[-2]
        prev_pos = (prev_point['latitude'], prev_point['longitude'])
        curr_pos = (last_point['latitude'], last_point['longitude'])
        distance_km = haversine(prev_pos, curr_pos, unit=Unit.KILOMETERS)
        
        if distance_km > 10:  # Impossible jump threshold
            violations.append(f'Impossible Jump ({distance_km:.1f} km)')
            rule_score += 0.40
    
    # R2: Suspected illegal fishing in MPA
    if last_point['speed'] < 3 and MARINE_PROTECTED_AREA.contains(current_position):
        violations.append('Suspected Illegal Fishing')
        rule_score += 0.35
    
    # R3: EEZ border loitering detection
    for zone, zone_name in [(NORWEGIAN_EEZ, 'Norwegian'), (SWEDISH_EEZ_BALTIC, 'Baltic')]:
        if not zone.contains(current_position):
            distance_to_border = zone.distance(current_position) * 111  # Convert to km
            if distance_to_border < 20:  # Within 20km of border
                violations.append(f'Loitering on {zone_name} EEZ Border')
                rule_score += 0.25
    
    # R4: Erratic speed behavior
    if len(sequence_df) > 2:
        recent_speeds = sequence_df['speed'].tail(3).values
        speed_variance = np.var(recent_speeds)
        if speed_variance > 25:  # High speed variance
            violations.append('Erratic Speed Pattern')
            rule_score += 0.20
    
    # R5: Sharp course changes
    if len(sequence_df) > 2:
        recent_courses = sequence_df['course'].tail(3).values
        course_diffs = np.diff(recent_courses) % 360
        max_turn = max(np.minimum(course_diffs, 360 - course_diffs))
        if max_turn > 45:  # Sharp turn threshold
            violations.append('Sharp Course Change')
            rule_score += 0.15
    
    return min(1.0, rule_score), violations

def calculate_hybrid_score(ml_decision, rule_score, rule_violations):
    """
    Calculate hybrid anomaly score using 70% ML + 30% Rules
    Returns: (is_anomaly, hybrid_score, reason_string)
    """
    # Normalize ML score to [0,1]
    ml_score_normalized = normalize_ml_score(ml_decision)
    
    # Calculate weighted hybrid score
    hybrid_score = (ML_WEIGHT * ml_score_normalized) + (RULE_WEIGHT * rule_score)
    
    # Determine anomaly status
    is_anomaly = hybrid_score > HYBRID_THRESHOLD
    
    # Generate human-readable reason
    reason_parts = []
    
    # Add ML component if significant
    if ml_score_normalized > 0.40:
        reason_parts.append(f"ML: Atypical Behavior (score: {ml_score_normalized:.2f})")
    
    # Add rule violations
    for violation in rule_violations:
        reason_parts.append(f"Rule: {violation}")
    
    # Final reason string
    if not reason_parts:
        reason = "Normal"
    else:
        reason = " | ".join(reason_parts)
    
    return is_anomaly, hybrid_score, reason

# --- Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'model_loaded': encoder is not None,
        'hybrid_weights': {'ml': ML_WEIGHT, 'rules': RULE_WEIGHT}
    })

# --- Main Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        start_time = time.time()
        
        # Extract and validate input data
        data = request.json.get('data')
        if not data or len(data) != SEQ_LEN:
            return jsonify({'error': f'Expected {SEQ_LEN} data points'}), 400
        
        # Convert to numpy array for ML processing
        sequence_array = np.array(data, dtype=np.float32)
        
        # ML Processing Pipeline
        scaled_sequence = scaler.transform(sequence_array).reshape(1, SEQ_LEN, len(FEATURES))
        latent_representation = encoder.predict(scaled_sequence, verbose=0)
        ml_decision = float(detector.decision_function(latent_representation)[0])
        
        # Rule-based Processing Pipeline
        sequence_df = pd.DataFrame(data, columns=FEATURES)
        rule_score, rule_violations = calculate_rule_score(sequence_df)
        
        # Hybrid Score Calculation
        is_anomaly, hybrid_score, reason = calculate_hybrid_score(ml_decision, rule_score, rule_violations)
        
        # Performance timing
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log prediction results
        vessel_mmsi = sequence_df.iloc[-1].get('mmsi', 'Unknown')
        if is_anomaly:
            logging.warning(f"ANOMALY DETECTED - MMSI: {vessel_mmsi} | Score: {hybrid_score:.3f} | Reason: {reason}")
        else:
            logging.info(f"Normal vessel behavior - MMSI: {vessel_mmsi} | Score: {hybrid_score:.3f}")
        
        # Return comprehensive prediction results
        return jsonify({
            'is_anomaly': bool(is_anomaly),
            'hybrid_score': round(hybrid_score, 3),
            'reason': reason,
            'scoring_breakdown': {
                'ml_decision': round(ml_decision, 4),
                'ml_score_normalized': round(normalize_ml_score(ml_decision), 3),
                'ml_weight': ML_WEIGHT,
                'rule_score': round(rule_score, 3),
                'rule_weight': RULE_WEIGHT,
                'rule_violations': rule_violations,
                'threshold': HYBRID_THRESHOLD
            },
            'performance': {
                'processing_time_ms': round(processing_time, 2)
            }
        })
        
    except Exception as e:
        logging.error(f"Prediction Error: {e}\n{traceback.format_exc()}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

if __name__ == '__main__':
    logging.info("🚢 Starting Enhanced Maritime Anomaly Detection API Server")
    logging.info(f"📊 Hybrid Scoring: {ML_WEIGHT*100}% ML + {RULE_WEIGHT*100}% Rules")
    logging.info(f"🎯 Anomaly Threshold: {HYBRID_THRESHOLD}")
    app.run(host='0.0.0.0', port=5002, debug=False)