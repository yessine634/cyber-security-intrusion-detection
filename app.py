from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load model and features
model = joblib.load('models/xgboost.pkl')
feature_names = joblib.load('models/model_features.pkl')

# Valid protocols from training data (133 unique)
VALID_PROTOCOLS = ['3pc', 'a/n', 'aes-sp3-d', 'any', 'argus', 'aris', 'arp', 'ax.25', 'bbn-rcc', 'bna', 
                   'br-sat-mon', 'cbt', 'cftp', 'chaos', 'compaq-peer', 'cphb', 'cpnx', 'crtp', 'crudp', 
                   'dcn', 'ddp', 'ddx', 'dgp', 'egp', 'eigrp', 'emcon', 'encap', 'etherip', 'fc', 'fire', 
                   'ggp', 'gmtp', 'gre', 'hmp', 'i-nlsp', 'iatp', 'ib', 'icmp', 'idpr', 'idpr-cmtp', 'idrp', 
                   'ifmp', 'igmp', 'igp', 'il', 'ip', 'ipcomp', 'ipcv', 'ipip', 'iplt', 'ipnip', 'ippc', 
                   'ipv6', 'ipv6-frag', 'ipv6-no', 'ipv6-opts', 'ipv6-route', 'ipx-n-ip', 'irtp', 'isis', 
                   'iso-ip', 'iso-tp4', 'kryptolan', 'l2tp', 'larp', 'leaf-1', 'leaf-2', 'merit-inp', 
                   'mfe-nsp', 'mhrp', 'micp', 'mobile', 'mtp', 'mux', 'narp', 'netblt', 'nsfnet-igp', 'nvp', 
                   'ospf', 'pgm', 'pim', 'pipe', 'pnni', 'pri-enc', 'prm', 'ptp', 'pup', 'pvp', 'qnx', 'rdp', 
                   'rsvp', 'rtp', 'rvd', 'sat-expak', 'sat-mon', 'sccopmce', 'scps', 'sctp', 'sdrp', 
                   'secure-vmtp', 'sep', 'skip', 'sm', 'smp', 'snp', 'sprite-rpc', 'sps', 'srp', 'st2', 
                   'stp', 'sun-nd', 'swipe', 'tcf', 'tcp', 'tlsp', 'tp++', 'trunk-1', 'trunk-2', 'ttp', 
                   'udp', 'unas', 'uti', 'vines', 'visa', 'vmtp', 'vrrp', 'wb-expak', 'wb-mon', 'wsn', 
                   'xnet', 'xns-idp', 'xtp', 'zero']

# State risk categorization function (same as in notebook)
def categorize_state_risk(state):
    # Critical Risk (> 80% Attack Rate)
    if state in ['INT', 'CLO']:
        return 'Critical'
    
    # High Risk (45% - 50% Attack Rate)
    elif state in ['FIN', 'ACC']:
        return 'High'
    
    # Medium Risk (14% - 17% Attack Rate)
    elif state in ['REQ', 'RST']:
        return 'Medium'
    
    # Low Risk (< 5% Attack Rate)
    else:
        return 'Low'  # Covers CON, ECO, PAR, URN, no 

# Risk mapping (same as in notebook)
risk_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2,
    'Critical': 3
}

def encode_state_risk(state):
    """Convert raw state to encoded risk value"""
    risk_category = categorize_state_risk(state)
    return risk_mapping[risk_category]

@app.route('/')
def home():
    return render_template('index.html', features=feature_names)

@app.route('/sample/<sample_type>')
def get_sample(sample_type):
    """Get sample data for testing - Normal or Attack"""
    try:
        df = pd.read_csv('Datasets/UNSW_NB15_testing-set.csv')
        
        if sample_type == 'normal':
            # Get a random normal sample
            samples = df[df['label'] == 0]
        else:
            # Get a random attack sample
            samples = df[df['label'] == 1]
        
        # Pick a random sample
        row = samples.sample(1).iloc[0]
        
        # Build the data dict with the features needed by the form
        sample_data = {}
        for feat in feature_names:
            if feat == 'state_risk_category_encoded':
                continue  # Skip - we'll use state instead
            elif feat in row.index:
                val = row[feat]
                # Convert numpy types to Python types
                if pd.isna(val):
                    sample_data[feat] = 0
                elif isinstance(val, (np.integer, np.floating)):
                    sample_data[feat] = float(val)
                else:
                    sample_data[feat] = str(val)
            else:
                sample_data[feat] = 0
        
        # Add state for the form
        sample_data['state'] = str(row['state'])
        sample_data['actual_label'] = 'Normal' if row['label'] == 0 else 'Attack'
        
        return jsonify(sample_data)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Debug: Print received data
        print("\n" + "="*50)
        print("RECEIVED DATA:")
        print(data)
        
        # Validate protocol
        proto = data.get('proto', 'tcp').lower()
        if proto not in VALID_PROTOCOLS:
            return jsonify({
                'error': f"Invalid protocol '{proto}'. Protocol not found in training data.",
                'valid_protocols': VALID_PROTOCOLS
            }), 400
        data['proto'] = proto
        
        # Extract state and convert to state_risk_category_encoded
        state = data.pop('state', 'FIN')  # Remove state from data
        data['state_risk_category_encoded'] = encode_state_risk(state)
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Ensure all features are present
        for col in feature_names:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_names]
        
        # Convert all columns to appropriate types
        # proto and service stay as strings (handled by pipeline's TargetEncoder)
        for col in input_df.columns:
            if col not in ['proto', 'service']:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0).astype(np.float64)
        
        # Ensure proto and service are strings
        if 'proto' in input_df.columns:
            input_df['proto'] = input_df['proto'].astype(str)
        if 'service' in input_df.columns:
            input_df['service'] = input_df['service'].astype(str)
        
        # Debug: Print final input to model
        print("\nFINAL INPUT TO MODEL:")
        for col in feature_names:
            print(f"  {col}: {input_df[col].values[0]}")
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        print(f"\nPREDICTION: {prediction} ({'Attack' if prediction == 1 else 'Normal'})")
        print(f"PROBABILITY: Normal={probability[0]:.4f}, Attack={probability[1]:.4f}")
        print("="*50)
        
        result = {
            'prediction': int(prediction),
            'label': 'Attack' if prediction == 1 else 'Normal',
            'confidence': float(max(probability) * 100),
            'prob_normal': float(probability[0] * 100),
            'prob_attack': float(probability[1] * 100),
            'state_risk': categorize_state_risk(state)
        }
        
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)