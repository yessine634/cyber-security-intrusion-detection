import pandas as pd
import joblib
import numpy as np
from sklearn.pipeline import Pipeline

# Load model and features
model = joblib.load('models/xgboost.pkl')
feature_names = joblib.load('models/model_features.pkl')

print("Model type:", type(model))
print("Pipeline steps:", model.named_steps if hasattr(model, 'named_steps') else "N/A")

# Check if it's a pipeline
if hasattr(model, 'named_steps'):
    encoder = model.named_steps.get('target_encoder')
    if encoder:
        print("\nTargetEncoder cols:", encoder.cols)
        print("TargetEncoder mapping (first few):")
        for col, mapping in encoder.mapping.items():
            print(f"  {col}: {dict(list(mapping['mapping'].items())[:5])}")

# Test accuracy on larger sample
df = pd.read_csv('Datasets/UNSW_NB15_testing-set.csv')

# Get predictions for all test data
correct_normal = 0
total_normal = 0
correct_attack = 0  
total_attack = 0

for _, row in df.head(500).iterrows():
    input_data = {}
    for feat in feature_names:
        if feat == 'state_risk_category_encoded':
            state = row['state']
            if state in ['INT', 'CLO']:
                input_data[feat] = 3
            elif state in ['FIN', 'ACC']:
                input_data[feat] = 2
            elif state in ['REQ', 'RST']:
                input_data[feat] = 1
            else:
                input_data[feat] = 0
        elif feat in row.index:
            input_data[feat] = row[feat]
        else:
            input_data[feat] = 0
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]
    
    for col in input_df.columns:
        if col not in ['proto', 'service']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0).astype(np.float64)
        else:
            input_df[col] = input_df[col].astype(str)
    
    pred = model.predict(input_df)[0]
    actual = row['label']
    
    if actual == 0:
        total_normal += 1
        if pred == 0:
            correct_normal += 1
    else:
        total_attack += 1
        if pred == 1:
            correct_attack += 1

print(f"\n=== ACCURACY ON 500 SAMPLES ===")
print(f"Normal: {correct_normal}/{total_normal} = {100*correct_normal/max(total_normal,1):.1f}%")
print(f"Attack: {correct_attack}/{total_attack} = {100*correct_attack/max(total_attack,1):.1f}%")
print(f"Overall: {correct_normal+correct_attack}/500 = {100*(correct_normal+correct_attack)/500:.1f}%")
