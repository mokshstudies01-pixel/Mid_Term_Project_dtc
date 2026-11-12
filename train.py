import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load & clean data
df = pd.read_csv("https://archive.ics.ucs.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv")
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[', '').str.replace(']', '')

# Targets
failure_cols = ['twf', 'hdf', 'pwf', 'osf', 'rnf']
def get_failure_type(row):
    if row['machine_failure'] == 0: return 'none'
    actives = [c for c in failure_cols if row[c] == 1]
    if len(actives) == 0: return 'other'
    return actives[0] if len(actives) == 1 else 'multi_failure'

df['failure_type'] = df.apply(get_failure_type, axis=1)

# Split features and target
num_cols = ['air_temperature_k', 'process_temperature_k', 'rotational_speed_rpm', 'torque_nm', 'tool_wear_min']
X = pd.get_dummies(df[num_cols + ['type']], drop_first=True)
y_bin = df['machine_failure']
y_multi = df['failure_type']

# Train final models on entire dataset
best_rf = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=150, class_weight='balanced', random_state=42)
best_rf.fit(X, y_bin)

rf_multi = RandomForestClassifier(n_estimators=150, min_samples_split=5, class_weight='balanced', random_state=42)
rf_multi.fit(X, y_multi)

# Export serialized models
with open('model_binary.bin', 'wb') as f: pickle.dump(best_rf, f)
with open('model_multiclass.bin', 'wb') as f: pickle.dump(rf_multi, f)
print("Standalone train.py finished successfully!")
