import os
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import argparse
import logging
from datetime import datetime

def setup_logger():

    os.makedirs("data", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler("data/log_file.log"),  
            logging.StreamHandler()  
        ]
    )
    return logging.getLogger()

logger = setup_logger()

class My_Classifier_Model:

    def __init__(self, model_type='xgboost', model_dir='models', results_dir='results'):

        self.model_type = model_type
        self.model_dir = model_dir
        self.results_dir = results_dir
        self.model = None

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Initialized model with type: {model_type}")

    # Заполнение пустых строк

    @staticmethod
    def remove_Nan_data(data):

        logger.info("Handling missing values...")

        numeric_data = data.select_dtypes(["int", "float"]).columns # Числовые параметры

        categorical_data = data.select_dtypes(exclude=["int", "float"]).columns # категориальные параметры

        for col in numeric_data:
            data[col] = data[col].fillna(data[col].median()).infer_objects(copy=False)

        for col in categorical_data:
            most_frequent = data[col].mode()[0]
            data[col] = data[col].fillna(most_frequent).infer_objects(copy=False)

        return data

    # Обработка данных

    @staticmethod
    def Data_transform(data): 

        logger.info("Transforming data...")

        cabin_data = data["Cabin"].str.split("/", expand=True)
        cabin_data.columns = ["Deck", "Num", "Side"]
        cabin_data["Num"] = cabin_data["Num"].fillna(-1).astype(int)
        cabin_data["Deck"] = cabin_data["Deck"].fillna("Unidentified") 
        cabin_data["Side"] = cabin_data["Side"].fillna("Unidentified")

        data["CryoSleep"] = data["CryoSleep"].astype(bool).fillna(False).astype(int)
        data["VIP"] = data["VIP"].astype(bool).fillna(False).astype(int)

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

    # Логистическая регрессия

    def train_logistic_regression(self, X_train, y_train):
        
        logger.info("Training Logistic Regression model...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        self.scaler = scaler  

        best_params = {
        'C': 1.9139784917677003,
        'class_weight': 'balanced',
        'fit_intercept': True,
        'intercept_scaling': 3.077722358806151,
        'max_iter': 2204,
        'solver': 'lbfgs',
        'tol': 0.002641694010428971,
        'warm_start': True
        }

        self.model = LogisticRegression (
        C=best_params['C'],
        class_weight=best_params['class_weight'],
        fit_intercept=best_params['fit_intercept'],
        intercept_scaling=best_params['intercept_scaling'],
        max_iter=best_params['max_iter'],
        solver=best_params['solver'],
        tol=best_params['tol'],
        warm_start=best_params['warm_start'],
        random_state=17,  
        n_jobs=-1 
    )
        self.model.fit(X_train_scaled, y_train)

    # XGB


    def train_xgboost(self, X_train, y_train):
        logger.info("Training XGBoost model...")
        self.model = XGBClassifier(
            booster='dart',
            reg_lambda =0.07286592647123674,
            alpha=1.2424664860821126,
            learning_rate=0.014461252660471866,
            n_estimators=767,
            max_depth=7,
            min_child_weight=2,
            gamma=0.20026700808078304,
            subsample=0.9937185065468269,
            colsample_bytree=0.7147580190620447,
            random_state=17,
            eval_metric='logloss'
        )
        self.model.fit(X_train, y_train)

    def train(self, data_path):

        logger.info(f"Starting training with dataset: {data_path}")
        data = pd.read_csv(data_path)
        data = self.remove_Nan_data(data)
        
        X = self.Data_transform(data)
        y = data['Transported'].astype('int')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.model_type == 'logistic_regression':
            self.train_logistic_regression(X_train, y_train)

        elif self.model_type == 'xgboost':
            self.train_xgboost(X_train, y_train)

        else:
            logger.error(f"Unknown model type: {self.model_type}")
            raise ValueError(f"Unknown model type: {self.model_type}")

        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, self.model.predict_proba(X_test)[:, 1])

        model_path = os.path.join(self.model_dir, "model.pkl")
        joblib.dump(self.model, model_path)

        logger.info(f"Model trained and saved to {model_path}")
        logger.info(f"Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")

    
    def predict(self, data_path):
        
        logger.info(f"Starting predictions with dataset: {data_path}")
        model_path = os.path.join(self.model_dir, "model.pkl")

        if not os.path.exists(model_path):
            logger.error(f"Model not found at {model_path}. Train the model first.")
            raise FileNotFoundError(f"Model not found at {model_path}. Train the model first.")

        self.model = joblib.load(model_path)

        data = pd.read_csv(data_path)
        data = self.remove_Nan_data(data)
        X = self.Data_transform(data)
        PassengerId = data['PassengerId']

        if self.model_type == 'logistic_regression':
            X_scaled = self.scaler.transform(X) 
            predictions = self.model.predict(X_scaled)

        else:
            predictions = self.model.predict(X)

        predictions = predictions.astype(bool)

        results_df = pd.DataFrame({
            'PassengerId': PassengerId,  
            'Transported': predictions    
        })
        
        results_path = os.path.join(self.results_dir, "predictions.csv")
        results_df.to_csv(results_path, index=False)

        logger.info(f"Predictions saved to {results_path}")



def main():

    parser = argparse.ArgumentParser(description="Train or predict using My_Classifier_Model.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--dataset", type=str, required=True, help="Path to the training dataset")
    train_parser.add_argument("--model_type", type=str, default="xgboost", help="Type of model to train (xgboost or logistic_regression)")

    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument("--dataset", type=str, required=True, help="Path to the evaluation dataset")
    predict_parser.add_argument("--model_type", type=str, default="xgboost", help="Type of model to use for predictions (xgboost or logistic_regression)")

    args = parser.parse_args()

    model = My_Classifier_Model(model_type=args.model_type)

    if args.command == "train":
        model.train(args.dataset)

    elif args.command == "predict":
        model.predict(args.dataset)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()