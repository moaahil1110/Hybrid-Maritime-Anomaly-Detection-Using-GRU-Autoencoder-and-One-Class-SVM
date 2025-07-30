# live_client.py - Enhanced AIS Data Streaming with Hybrid Scoring Integration
import websocket
import json
import time
import requests
import logging
import threading
from flask import Flask
from flask_socketio import SocketIO
from flask_cors import CORS

# --- Enhanced Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - Client - %(levelname)s - %(message)s')
# Suppress verbose framework logs for cleaner output
logging.getLogger('socketio').setLevel(logging.ERROR)
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# --- Configuration ---
AISSTREAM_API_KEY = "f21c889c57c775dfecd998de7b0d5ee7ef823edb"
MODEL_API_URL = "http://127.0.0.1:5002/predict"
HEALTH_CHECK_URL = "http://127.0.0.1:5002/health"
SEQ_LEN = 10
FEATURES = ['latitude', 'longitude', 'speed', 'course', 'heading']
BBOX_CORNERS = [[[54.9, -0.5], [74.6, 36.5]]]  # Norwegian/Baltic Sea region

# --- Setup Flask-SocketIO Server for GUI Communication ---
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connections
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Global State Tracking ---
vessel_tracks = {}
active_connections = 0
prediction_stats = {
    'total_predictions': 0,
    'anomalies_detected': 0,
    'vessels_tracked': 0,
    'api_errors': 0
}

# --- SocketIO Event Handlers ---
@socketio.on('connect')
def handle_gui_connect():
    global active_connections
    active_connections += 1
    logging.info(f"🖥️  GUI Client connected. Active connections: {active_connections}")

@socketio.on('disconnect')
def handle_gui_disconnect():
    global active_connections
    active_connections -= 1
    logging.info(f"🖥️  GUI Client disconnected. Active connections: {active_connections}")

def check_api_health():
    """Verify that the prediction API is running"""
    try:
        response = requests.get(HEALTH_CHECK_URL, timeout=5)
        if response.status_code == 200:
            logging.info("✅ Prediction API health check passed")
            return True
    except:
        pass
    logging.warning("⚠️  Prediction API health check failed")
    return False

def process_vessel_data(ais_point):
    """
    Process incoming AIS data point and handle prediction pipeline
    """
    mmsi = ais_point.get('mmsi')
    if not mmsi:
        return
    
    # Initialize vessel tracking if new
    if mmsi not in vessel_tracks:
        vessel_tracks[mmsi] = []
        prediction_stats['vessels_tracked'] += 1
        logging.info(f"📡 New vessel tracked: MMSI {mmsi}")
    
    # Add point to vessel track
    vessel_tracks[mmsi].append(ais_point)
    
    # Maintain sliding window of SEQ_LEN points
    if len(vessel_tracks[mmsi]) > SEQ_LEN:
        vessel_tracks[mmsi].pop(0)
    
    # Emit immediate position update for GUI responsiveness
    socketio.emit('vessel_position', {
        'mmsi': mmsi,
        'point': ais_point,
        'track_length': len(vessel_tracks[mmsi]),
        'timestamp': time.time()
    })
    
    # Process prediction when we have enough data points
    if len(vessel_tracks[mmsi]) == SEQ_LEN:
        try:
            # Prepare sequence for ML API
            sequence_for_prediction = [
                [point[feature] for feature in FEATURES] 
                for point in vessel_tracks[mmsi]
            ]
            
            # Call hybrid prediction API
            start_time = time.time()
            response = requests.post(
                MODEL_API_URL, 
                json={'data': sequence_for_prediction},
                timeout=10
            )
            api_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                prediction_result = response.json()
                prediction_stats['total_predictions'] += 1
                
                # Track anomaly statistics
                if prediction_result.get('is_anomaly', False):
                    prediction_stats['anomalies_detected'] += 1
                    logging.warning(
                        f"🚨 ANOMALY DETECTED - MMSI {mmsi} | "
                        f"Score: {prediction_result.get('hybrid_score', 0):.3f} | "
                        f"Reason: {prediction_result.get('reason', 'Unknown')}"
                    )
                else:
                    logging.info(f"✅ Normal behavior - MMSI {mmsi} | Score: {prediction_result.get('hybrid_score', 0):.3f}")
                
                # Broadcast comprehensive update to GUI
                socketio.emit('update', {
                    'mmsi': mmsi,
                    'point': ais_point,
                    'is_anomaly': prediction_result.get('is_anomaly', False),
                    'hybrid_score': prediction_result.get('hybrid_score', 0),
                    'reason': prediction_result.get('reason', 'Processing...'),
                    'scoring_breakdown': prediction_result.get('scoring_breakdown', {}),
                    'performance': {
                        'api_response_time_ms': round(api_time, 2)
                    },
                    'timestamp': time.time()
                })
                
            else:
                prediction_stats['api_errors'] += 1
                logging.error(f"❌ API Error for MMSI {mmsi}: HTTP {response.status_code}")
                
                # Send error notification to GUI
                socketio.emit('prediction_error', {
                    'mmsi': mmsi,
                    'error': f'API returned status {response.status_code}',
                    'timestamp': time.time()
                })
                
        except requests.exceptions.ConnectionError:
            prediction_stats['api_errors'] += 1
            logging.error("❌ Cannot connect to prediction API. Ensure app.py is running on port 5002")
            
        except requests.exceptions.Timeout:
            prediction_stats['api_errors'] += 1
            logging.error(f"⏱️  API timeout for MMSI {mmsi}")
            
        except Exception as e:
            prediction_stats['api_errors'] += 1
            logging.error(f"❌ Prediction processing error for MMSI {mmsi}: {e}")

# --- AISStream WebSocket Event Handlers ---
def on_aisstream_message(ws, message):
    """Handle incoming AIS messages from AISStream.io"""
    try:
        ais_message = json.loads(message)
        
        # Process position reports
        if ais_message.get('MessageType') == 'PositionReport':
            position_data = ais_message['Message']['PositionReport']
            
            # Extract and normalize vessel data
            vessel_point = {
                'mmsi': position_data.get('UserID'),
                'latitude': float(position_data.get('Latitude', 0.0)),
                'longitude': float(position_data.get('Longitude', 0.0)),
                'speed': float(position_data.get('Sog', 0.0)),  # Speed over ground
                'course': float(position_data.get('Cog', 0.0)),  # Course over ground
                'heading': position_data.get('TrueHeading', 511)
            }
            
            # Handle invalid heading (511 = not available)
            if vessel_point['heading'] == 511:
                vessel_point['heading'] = vessel_point['course']
            else:
                vessel_point['heading'] = float(vessel_point['heading'])
            
            # Validate data quality
            if (vessel_point['mmsi'] and 
                -90 <= vessel_point['latitude'] <= 90 and 
                -180 <= vessel_point['longitude'] <= 180):
                
                process_vessel_data(vessel_point)
            else:
                logging.debug(f"⚠️  Invalid AIS data: {vessel_point}")
                
    except json.JSONDecodeError:
        logging.error("❌ Invalid JSON in AIS message")
    except Exception as e:
        logging.error(f"❌ AIS message processing error: {e}")

def on_aisstream_error(ws, error):
    """Handle AISStream WebSocket errors"""
    if not isinstance(error, websocket.WebSocketConnectionClosedException):
        logging.error(f"🌐 AISStream WebSocket Error: {error}")

def on_aisstream_close(ws, close_status_code, close_msg):
    """Handle AISStream connection closure"""
    logging.warning(f"🌐 AISStream connection closed - Code: {close_status_code}, Message: {close_msg}")

def on_aisstream_open(ws):
    """Handle AISStream connection opening"""
    logging.info("🌐 AISStream WebSocket connection established")
    
    # Subscribe to AIS data for specified region
    subscription_message = {
        "APIKey": AISSTREAM_API_KEY,
        "BoundingBoxes": BBOX_CORNERS,
        "FilterMessageTypes": ["PositionReport"]  # Only position reports
    }
    
    ws.send(json.dumps(subscription_message))
    logging.info(f"📡 Subscribed to AIS data for region: {BBOX_CORNERS}")

def run_aisstream_client():
    """Main AISStream client loop with reconnection logic"""
    reconnect_delay = 10
    max_reconnect_delay = 300  # 5 minutes max
    
    while True:
        try:
            logging.info("🔄 Connecting to AISStream.io...")
            
            websocket_app = websocket.WebSocketApp(
                "wss://stream.aisstream.io/v0/stream",
                on_open=on_aisstream_open,
                on_message=on_aisstream_message,
                on_error=on_aisstream_error,
                on_close=on_aisstream_close
            )
            
            websocket_app.run_forever()
            
        except Exception as e:
            logging.error(f"❌ AISStream client error: {e}")
        
        logging.info(f"⏱️  Reconnecting to AISStream in {reconnect_delay} seconds...")
        time.sleep(reconnect_delay)
        
        # Exponential backoff for reconnection delay
        reconnect_delay = min(reconnect_delay * 1.5, max_reconnect_delay)

# --- Status Endpoint for Monitoring ---
@app.route('/status', methods=['GET'])
def get_status():
    """Provide system status information"""
    return {
        'active_gui_connections': active_connections,
        'vessels_being_tracked': len(vessel_tracks),
        'prediction_statistics': prediction_stats,
        'api_health': check_api_health(),
        'timestamp': time.time()
    }

# --- Main Application Entry Point ---
if __name__ == "__main__":
    logging.info("🚢 Starting Enhanced Maritime Anomaly Detection Streaming Server")
    logging.info(f"📊 Target API: {MODEL_API_URL}")
    logging.info(f"🌐 AIS Data Region: {BBOX_CORNERS}")
    
    # Verify API connectivity before starting
    if not check_api_health():
        logging.warning("⚠️  Warning: Prediction API not responding. Ensure app.py is running.")
    
    # Start AISStream client in background thread
    ais_thread = threading.Thread(target=run_aisstream_client, daemon=True)
    ais_thread.start()
    logging.info("🚀 AISStream client thread started")
    
    # Start Flask-SocketIO server for GUI communication
    try:
        logging.info("🖥️  Starting GUI streaming server on http://127.0.0.1:5003")
        socketio.run(app, host='0.0.0.0', port=5003, debug=False)
    except Exception as e:
        logging.error(f"❌ Failed to start GUI server: {e}")
        exit(1)
