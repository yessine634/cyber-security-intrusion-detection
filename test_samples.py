import pandas as pd
import joblib
import numpy as np

# Load model and features
model = joblib.load('models/xgboost.pkl')
feature_names = joblib.load('models/model_features.pkl')

# Load testing set
df = pd.read_csv('Datasets/UNSW_NB15_testing-set.csv')

print("="*60)
print("TESTING MODEL ON 1000 SAMPLES")
print("="*60)

correct = 0
total = 0
normal_correct = 0
normal_total = 0
attack_correct = 0
attack_total = 0

# Test first 1000 samples
for idx, row in df.head(1000).iterrows():
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
    
    input_df = pd.DataFrame([input_data])[feature_names]
    for col in input_df.columns:
        if col not in ['proto', 'service']:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0).astype(np.float64)
        else:
            input_df[col] = input_df[col].astype(str)
    
    pred = model.predict(input_df)[0]
    actual = row['label']
    
    total += 1
    if pred == actual:
        correct += 1
    
    if actual == 0:
        normal_total += 1
        if pred == 0:
            normal_correct += 1
    else:
        attack_total += 1
        if pred == 1:
            attack_correct += 1

print(f"\nTotal samples: {total}")
print(f"Overall accuracy: {100*correct/total:.2f}%")
print(f"\nNormal samples: {normal_total}")
print(f"Normal accuracy: {100*normal_correct/normal_total:.2f}% ({normal_correct}/{normal_total})")
print(f"\nAttack samples: {attack_total}")
print(f"Attack accuracy: {100*attack_correct/attack_total:.2f}% ({attack_correct}/{attack_total})")
