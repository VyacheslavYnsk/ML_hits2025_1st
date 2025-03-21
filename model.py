import os
import pandas as pd
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from IPython.display import FileLink
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

import joblib

class My_Classifier_Model:
    
    def __init__(self, model_type=''):
        
        data = pd.read_csv('/Users/vaceslav/Ml1/Origin/train.csv')
        data_test = pd.read_csv('/Users/vaceslav/Ml1/Origin/test.csv')
        sample_submission = pd.read_csv("/Users/vaceslav/Ml1/Origin/sample_submission.csv")
        sample_submission_path = "/Users/vaceslav/Ml1/Origin/sample_submission.csv"

        data = remove_Nan_data(data)
        data_test = remove_Nan_data(data_test)
        X = Data_transform(data)
        y = data['Transported'].astype('int')
        X_test = Data_transform(data_test)
    
    def remove_Nan_data(data):
        numeric_data = data.select_dtypes(["int", "float"]).columns
        categorical_data = data.select_dtypes(exclude=["int", "float"]).columns

        for col in numeric_data:
            data[col] = data[col].fillna(data[col].median())

        for col in categorical_data:
            most_frequent = data[col].mode()[0]  
            data[col] = data[col].astype("object")  
            data[col] = data[col].fillna(most_frequent)
            data[col] = data[col].infer_objects(copy=False)  

        return data
        
    
    def Data_transform(data):

        cabin_data = data["Cabin"].str.split("/", expand=True)
        cabin_data.columns = ["Deck", "Num", "Side"]
        cabin_data["Num"] = cabin_data["Num"].fillna(-1).astype(int)
        cabin_data["Deck"] = cabin_data["Deck"].fillna("Unidentified")
        cabin_data["Side"] = cabin_data["Side"].fillna("Unidentified")
    
        data["CryoSleep"] = data["CryoSleep"].astype(bool).fillna(False).astype(int)
        data["VIP"] = data["CryoSleep"].astype(bool).fillna(False).astype(int)


        spends_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
        data["Amount"] = data[spends_columns].sum(axis=1)


        numeric_columns = [
            'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', "Amount", "Age", "CryoSleep", "VIP"
        ]
    
        numerics = data.copy()[numeric_columns]
        columns_for_dummies = ["HomePlanet", "Destination"]
        dummies = pd.get_dummies(cabin_data[["Deck", "Side"]].join(data.copy()[columns_for_dummies]))
        dummies = dummies.astype(int)

        result = pd.concat([numerics, dummies], axis=1)

        return result
    
    def objective_lr(trial, X, y):

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=17, stratify=y)

        C = trial.suggest_loguniform("C", 1e-3, 10.0)  
        max_iter = trial.suggest_int("max_iter", 100, 10000)  
        solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"])  
        tol = trial.suggest_loguniform("tol", 1e-5, 1e-1)  # Порог сходимости
        fit_intercept = trial.suggest_categorical("fit_intercept", [True, False])  
        intercept_scaling = trial.suggest_loguniform("intercept_scaling", 0.1, 10.0)  
        class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])  
        warm_start = trial.suggest_categorical("warm_start", [True, False])  

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)

        model = LogisticRegression(
            C=C, max_iter=max_iter, solver=solver, tol=tol,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
            class_weight=class_weight, warm_start=warm_start,
            random_state=17, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        valid_pred = model.predict_proba(X_valid_scaled)[:, 1]
        auc_score = roc_auc_score(y_valid, valid_pred)

        return auc_score  

    def train(self, data_path):
       
        data = pd.read_csv(data_path)
        X = data.drop('target', axis=1)  
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])

        model_path = os.path.join(self.model_dir, "model.pkl")
        joblib.dump(self.model, model_path)

        print(f"Model trained and saved to {model_path}")
        print(f"Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

    
    def get_auc_lr_valid(X, y, data_test, sample_submission_path, best_params, ratio=0.7, seed=17):
        train_len = int(ratio * X.shape[0])
        X_train, X_valid = X.iloc[:train_len, :], X.iloc[train_len:, :]
        y_train, y_valid = y.iloc[:train_len], y.iloc[train_len:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(data_test)

        logit = LogisticRegression(**best_params)
        
        logit.fit(X_train_scaled, y_train)

        valid_pred = logit.predict_proba(X_valid_scaled)[:, 1]
        auc_score = roc_auc_score(y_valid, valid_pred)

        precision, recall, thresholds = precision_recall_curve(y_valid, valid_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_threshold = thresholds[np.argmax(f1_scores)]

        test_pred = logit.predict_proba(X_test_scaled)[:, 1]
        test_pred_labels = test_pred >= best_threshold

        sample_submission = pd.read_csv(sample_submission_path)
        sample_submission["Transported"] = test_pred_labels.astype(bool)
        sample_submission.to_csv("sample_submission.csv", index=False)

        return auc_score, FileLink("sample_submission.csv")
    

    def predict(self, data_path):
        
        model_path = os.path.join(self.model_dir, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

        self.model = joblib.load(model_path)

        data = pd.read_csv(data_path)
        X = data  

        predictions = self.model.predict(X)

        results = pd.DataFrame({'predictions': predictions})
        results.to_csv(self.results_dir, index=False)

        print(f"Predictions saved to {self.results_dir}")
    
    def objective_xgb(trial, X, y, sample_submission_path, X_test):

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.3, random_state=17, stratify=y
        )

        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'lambda': trial.suggest_float('lambda', 1e-3, 10.0, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10.0, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 1e-1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 1e-3, 10.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }

        model = XGBClassifier(**param) 
        model.fit(X_train, y_train)

        valid_pred = model.predict_proba(X_valid)[:, 1]
        auc_score = roc_auc_score(y_valid, valid_pred)

        return auc_score  
    
    def get_auc_xgb_valid(X, y, sample_submission_path, X_test, best_trial_xgb):

        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=0.21, random_state=17, stratify=y
        )

        best_params = {
            'booster': 'dart',
            'lambda': 0.07286592647123674,
            'alpha': 1.2424664860821126,
            'learning_rate': 0.014461252660471866,
            'n_estimators': 767,
            'max_depth': 7,
            'min_child_weight': 2,
            'gamma': 0.20026700808078304,
            'subsample': 0.9937185065468269,
            'colsample_bytree': 0.7147580190620447
        }

        model = XGBClassifier(**best_params, random_state=17, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)

        valid_pred = model.predict_proba(X_valid)[:, 1]
        auc_score = roc_auc_score(y_valid, valid_pred)

        test_pred = model.predict_proba(X_test)[:, 1]
        test_pred_labels = test_pred >= 0.5  

        sample_submission = pd.read_csv(sample_submission_path)
        sample_submission["Transported"] = test_pred_labels.astype(bool)
        sample_submission.to_csv("sample_submission.csv", index=False)

        return auc_score, FileLink("sample_submission.csv")

if __name__ == "__main__":
    model = My_Classifier_Model(model_type='random_forest')

    model.train("data/train.csv")

    model.predict("data/test.csv")