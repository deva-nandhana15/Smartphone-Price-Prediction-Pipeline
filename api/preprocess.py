"""
Preprocessing utilities used by the API. Must be consistent with Spark preprocessing.
Load metadata.json and scaler.joblib and transform incoming JSON to model input vector.
"""

import json
import joblib
import numpy as np
import os

class Preprocessor:
    def __init__(self, metadata_path, scaler_path=None):
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        self.feature_columns = self.metadata['feature_columns']
        self.categorical_mappings = self.metadata.get('categorical_mappings', {})
        self.bool_columns = self.metadata.get('bool_columns', [])
        self.scaler = None
        if scaler_path and os.path.exists(scaler_path):
            try:
                self.scaler = joblib.load(scaler_path)
            except Exception:
                self.scaler = None

    def preprocess_single(self, payload: dict):
        """
        payload: dict containing fields similar to the dataset
        returns: numpy array of feature vector (1D)
        """
        row = []
        for c in self.feature_columns:
            if c in self.categorical_mappings:
                mapping = self.categorical_mappings[c]
                val = payload.get(c)
                mapped = mapping.get(val, mapping.get('__OTHER__'))
                row.append(float(mapped if mapped is not None else mapping.get('__OTHER__', 0)))
            elif c in self.bool_columns:
                val = payload.get(c)
                if isinstance(val, bool):
                    row.append(1.0 if val else 0.0)
                elif isinstance(val, (int, float)):
                    row.append(1.0 if val else 0.0)
                elif isinstance(val, str):
                    row.append(1.0 if val.lower() in ('true','1','yes') else 0.0)
                else:
                    row.append(0.0)
            else:
                # numeric
                val = payload.get(c)
                try:
                    row.append(float(val) if val is not None and val != "" else 0.0)
                except Exception:
                    row.append(0.0)
        arr = np.array(row, dtype=np.float32).reshape(1, -1)
        if self.scaler is not None:
            try:
                arr = self.scaler.transform(arr)
            except Exception:
                pass
        return arr
