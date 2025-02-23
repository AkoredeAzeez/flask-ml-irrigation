import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
file_path = "data/irrigation_strategy_with_soil_type.csv"

class IrrigationSystem:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.label_encoders = {}
        self.label_encoder_strategy = LabelEncoder()
        self.X_train, self.X_test, self.y_train_class, self.y_test_class = None, None, None, None
        self.y_train_reg, self.y_test_reg = None, None
        self.classifier = RandomForestClassifier(random_state=42)
        self.regressor = RandomForestRegressor(random_state=42)
        self.load_and_preprocess_data()
        self.train_models()
    
    def load_and_preprocess_data(self):
        """Load and preprocess dataset."""
        self.df = pd.read_csv(self.file_path)
        
        # Encode categorical variables
        categorical_cols = ["crop", "season", "altitude", "soil_type"]
        self.label_encoders = {col: LabelEncoder() for col in categorical_cols}
        
        for col in categorical_cols:
            self.df[col] = self.label_encoders[col].fit_transform(self.df[col])
        
        # Encode target variable for classification (irrigation_strategy)
        self.df["irrigation_strategy"] = self.label_encoder_strategy.fit_transform(self.df["irrigation_strategy"])
        
        # Define features and targets
        X = self.df.drop(columns=["irrigation_strategy", "total_water_requirement_m3", "water_requirement_mm_day"])
        y_classification = self.df["irrigation_strategy"]
        y_regression = self.df["water_requirement_mm_day"]
        
        # Split into train and test sets
        self.X_train, self.X_test, self.y_train_class, self.y_test_class = train_test_split(
            X, y_classification, test_size=0.2, random_state=42
        )
        self.y_train_reg, self.y_test_reg = train_test_split(
            y_regression, test_size=0.2, random_state=42
        )
    
    def train_models(self):
        """Train classification and regression models."""
        self.classifier.fit(self.X_train, self.y_train_class)
        self.regressor.fit(self.X_train, self.y_train_reg)
    
    def evaluate_models(self):
        """Evaluate model performance."""
        y_pred_class = self.classifier.predict(self.X_test)
        y_pred_reg = self.regressor.predict(self.X_test)
        
        classification_accuracy = accuracy_score(self.y_test_class, y_pred_class)
        regression_mse = mean_squared_error(self.y_test_reg, y_pred_reg)
        
        return {
            "classification_accuracy": classification_accuracy,
            "regression_mse": regression_mse
        }
    
    def get_data_summary(self):
        """Return dataset summary for Flask app."""
        return {
            "columns": list(self.df.columns),
            "sample_data": self.df.head().to_dict(orient='records')
        }


