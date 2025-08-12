import numpy as np
import pandas as pd
import pickle
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import ast


class LinearRegressor:
    def __init__(self, target_variable):
        """
        Initialize the LinearRegressor.

        Parameters:
        target_variable (str): The name of the target variable ('ai_pred' or 'huma_pred')
        """
        self.target_variable = target_variable
        self.model = None
        self.means = {}
        self.stds = {}
        self.features = []

    def train(self, data):
        """
        Train the linear regression model.

        Parameters:
        data (pd.DataFrame): The training data containing the features and target variable.
        """
        # Copy data to avoid modifying the original dataset
        data = data.copy()

        # Standardize continuous variables
        continuous_vars = ['temperature', 'wind_speed', 'visibility', 'precipitation']
        for var in continuous_vars:
            mean = data[var].mean()
            std = data[var].std()
            if std == 0:
                std = 1  # Avoid division by zero
            self.means[var] = mean
            self.stds[var] = std
            data[f'{var}_std'] = (data[var] - mean) / std

        # Convert 'mines' to binary (0 for no mine, 1 for mine present)
        data['mines_binary'] = data['mines'] - 1

        # Prepare feature matrix X and target variable y
        feature_cols = [f'{var}_std' for var in continuous_vars] + ['mines_binary']
        X = data[feature_cols]
        y = data[self.target_variable]

        # Add constant term for intercept
        X = sm.add_constant(X)
        self.features = X.columns.tolist()

        # Train the model using statsmodels OLS
        self.model = sm.OLS(y, X).fit()
        print(self.model.summary())

    def evaluate(self, data):
        """
        Evaluate the model on the provided data.

        Parameters:
        data (pd.DataFrame): The evaluation data containing the features and target variable.

        Returns:
        rmse (float): The root mean squared error.
        """
        # Copy data to avoid modifying the original dataset
        data = data.copy()

        # Standardize continuous variables
        continuous_vars = ['temperature', 'wind_speed', 'visibility', 'precipitation']
        for var in continuous_vars:
            data[f'{var}_std'] = (data[var] - self.means[var]) / self.stds[var]

        # Convert 'mines' to binary
        data['mines_binary'] = data['mines'] - 1

        # Prepare feature matrix X and target variable y_true
        feature_cols = [f'{var}_std' for var in continuous_vars] + ['mines_binary']
        X = data[feature_cols]
        y_true = data[self.target_variable]

        # Add constant term for intercept
        X = sm.add_constant(X)

        # Predict using the model
        y_pred = self.model.predict(X)

        # Compute RMSE
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))
        return rmse

    def save_model(self, filepath):
        """
        Save the model and standardization parameters to a file.

        Parameters:
        filepath (str): The path to the file where the model will be saved.
        """
        # Save the model and parameters using pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'means': self.means,
                'stds': self.stds,
                'features': self.features
            }, f)

    def load_model(self, filepath):
        """
        Load the model and standardization parameters from a file.

        Parameters:
        filepath (str): The path to the file from which the model will be loaded.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.means = data['means']
            self.stds = data['stds']
            self.features = data['features']

    def predict(self, input_df):
        """
        Predict the target variable based on the input DataFrame.

        Parameters:
        input_df (pd.DataFrame): A DataFrame containing the feature values.

        Returns:
        predictions (np.ndarray): The predicted values of the target variable, clipped between 0 and 1.
        """
        # Ensure that input_df is a copy to prevent modifying the original data
        X = input_df.copy()

        # Required columns
        required_columns = ['temperature', 'wind_speed', 'visibility', 'precipitation', 'mines']

        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in input DataFrame: {missing_cols}")

        # Standardize continuous variables
        for var in ['temperature', 'wind_speed', 'visibility', 'precipitation']:
            X[f'{var}_std'] = (X[var] - self.means[var]) / self.stds[var]

        # Convert 'mines' to binary
        X['mines_binary'] = X['mines'] - 1

        # Prepare feature columns excluding 'const'
        feature_cols = [col for col in self.features if col != 'const']

        # Select the standardized features
        X = X[feature_cols]

        # Add constant term
        X = sm.add_constant(X, has_constant='add')

        # Reorder columns to match the training features
        X = X[self.features]

        # Predict using the model
        predictions = self.model.predict(X)

        # Clip predictions between 0 and 1
        predictions_clipped = np.clip(predictions, 0, 1)
        return predictions_clipped


if __name__ == '__main__':
    regressor_ai = LinearRegressor(target_variable='ai_pred')

    file_path = "data/save.csv"
    data = pd.read_csv(file_path)

    # Convert 'metadata' string representation to actual dictionary
    data['metadata'] = data['metadata'].apply(ast.literal_eval)
    # Extract metadata fields into separate columns
    metadata_df = pd.json_normalize(data['metadata'])
    data = pd.concat([data.drop(columns=['metadata']), metadata_df], axis=1)
    data['terrain_encoded'] = data['terrain'].astype('category').cat.codes

    # Split data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # Train the model for 'ai_pred'
    regressor_ai.train(train_data)

    # Evaluate the model
    rmse_ai = regressor_ai.evaluate(test_data)
    print(f"RMSE for 'ai_pred' on test data: {rmse_ai}")

    # Save the model
    regressor_ai.save_model('prediction/ai_regressor_model.pkl')

    # Train the model for 'huma_pred'
    regressor_huma = LinearRegressor(target_variable='huma_pred')
    regressor_huma.train(train_data)

    # Evaluate the model
    rmse_huma = regressor_huma.evaluate(test_data)
    print(f"RMSE for 'huma_pred' on test data: {rmse_huma}")

    # Save the model
    regressor_huma.save_model('prediction/huma_regressor_model.pkl')
