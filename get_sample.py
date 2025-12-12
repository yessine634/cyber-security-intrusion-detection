import pandas as pd
import joblib

# Load feature names
feature_names = joblib.load('models/model_features.pkl')

# Load test data
df = pd.read_csv('Datasets/UNSW_NB15_testing-set.csv')

# Get a normal sample
normal = df[df['label'] == 0].iloc[5]

print('NORMAL TRAFFIC SAMPLE - Use these values in the form:')
print('='*60)
for feat in feature_names:
    if feat in normal.index:
        print(f'{feat}: {normal[feat]}')
    elif feat == 'state_risk_category_encoded':
        print(f'{feat}: (from state)')
print('='*60)
print(f"state: {normal['state']}")
print(f"label: {normal['label']} (should be 0=Normal)")

print('\n\nATTACK TRAFFIC SAMPLE:')
print('='*60)
attack = df[df['label'] == 1].iloc[5]
for feat in feature_names:
    if feat in attack.index:
        print(f'{feat}: {attack[feat]}')
    elif feat == 'state_risk_category_encoded':
        print(f'{feat}: (from state)')
print('='*60)
print(f"state: {attack['state']}")
print(f"label: {attack['label']} (should be 1=Attack)")
