import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.metrics import precision_recall_curve, auc, classification_report
from imblearn.pipeline import make_pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import joblib

# ----------------------------
# Data Loading & Preprocessing
# ----------------------------
def load_data(path):
    """Load and preprocess transactional data"""
    raw_data = pd.read_parquet(path)
    
    # Temporal feature engineering
    raw_data['transaction_hour'] = raw_data['timestamp'].dt.hour
    raw_data['days_since_last_txn'] = raw_data.groupby('user_id')['timestamp'].diff().dt.days.fillna(30)
    
    # Monetary feature scaling
    raw_data['amount'] = RobustScaler().fit_transform(
        raw_data[['amount']]
    )
    
    return raw_data.drop(columns=['timestamp'])

data = load_data("your_transaction_data.parquet")

# ----------------------------
# Temporal Validation Split
# ----------------------------
# Time-based data partitioning
train_data = data[data['transaction_date'] < '2024-06-01']
test_data = data[data['transaction_date'] >= '2024-06-01']

X_train = train_data.drop(columns=['fraud_label', 'transaction_id'])
y_train = train_data['fraud_label']
X_test = test_data.drop(columns=['fraud_label', 'transaction_id']) 
y_test = test_data['fraud_label']

# ----------------------------
# Hybrid Ensemble Model Pipeline
# ----------------------------
# Feature transformation pipeline
preprocessor = make_pipeline(
    FunctionTransformer(validate_features),  # Add custom feature checks
    SMOTE(sampling_strategy=0.25, k_neighbors=5),
)

# Multi-algorithm ensemble
model = VotingClassifier(estimators=[
    ('xgb', XGBClassifier(
        scale_pos_weight=45,
        max_depth=7,
        learning_rate=0.1
    )),
    ('rf', RandomForestClassifier(
        class_weight='balanced',
        n_estimators=200,
        max_features='sqrt'
    )),
    ('lgbm', LGBMClassifier(
        is_unbalance=True,
        num_leaves=31,
        feature_fraction=0.8
    ))
], voting='soft', n_jobs=-1)

# Full processing pipeline
pipeline = make_pipeline(preprocessor, model)

# ----------------------------
# Model Training
# ----------------------------
pipeline.fit(X_train, y_train)

# ----------------------------
# Evaluation & Monitoring
# ----------------------------
def evaluate_model(model, X, y):
    """Advanced fraud-specific evaluation"""
    y_prob = model.predict_proba(X)[:,1]
    precision, recall, _ = precision_recall_curve(y, y_prob)
    auprc = auc(recall, precision)
    
    print(f"AUPRC: {auprc:.3f}")
    print(classification_report(y, model.predict(X)))
    
    # Monitor feature drift
    drift_report = calculate_feature_drift(X_train, X_test)
    print("Feature Drift Analysis:\n", drift_report)

evaluate_model(pipeline, X_test, y_test)

# ----------------------------
# Model Deployment
# ----------------------------
joblib.dump(pipeline, 'fraud_model_v1.pkl')

# Real-time inference template
class FraudDetector:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)
        self.feature_store = RealTimeFeatureStore()
    
    def predict(self, transaction):
        """Process transaction with feature enrichment"""
        enriched_data = self.feature_store.enrich(transaction)
        return self.model.predict_proba(enriched_data)[:,1]
