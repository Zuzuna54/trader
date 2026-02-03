import xgboost as xgb
import joblib
import os
import logging

class XGBoostModel:
    def __init__(self, model_path='models/xgboost_btc.json'):
        self.model_path = model_path
        self.model = None

    def train(self, X, y):
        logging.info("Training XGBoost model...")
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42
        )
        self.model.fit(X, y)
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)
        logging.info(f"Model saved to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)
            logging.info("Model loaded.")
            return True
        return False

    def predict(self, X):
        if self.model is None:
            if not self.load():
                raise Exception("Model not trained or loaded.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if self.model is None:
            if not self.load():
                raise Exception("Model not trained or loaded.")
        return self.model.predict_proba(X)
