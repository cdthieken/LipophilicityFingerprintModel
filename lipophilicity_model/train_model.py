import os
import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Define the path to the dataset
path = "lipophilicity_model/Lipophilicity.csv"  # Adjust path if necessary

# Check if the dataset exists
if not os.path.exists(path):
    raise FileNotFoundError(f"The dataset was not found at the path: {path}")

# Create a Morgan fingerprint generator
morgan_generator = GetMorganGenerator(radius=2, fpSize=1024)

# Function to generate Morgan fingerprints
def generate_morgan_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(morgan_generator.GetFingerprint(mol))
    else:
        return None

# Main function
def main(n_estimators, max_depth):
    # Load the data
    lipo_data = pd.read_csv(path)

    # Generate Morgan fingerprints
    lipo_data['morgan_fp'] = lipo_data['smiles'].apply(generate_morgan_fingerprint)

    # Drop rows with missing fingerprints
    lipo_data = lipo_data.dropna(subset=['morgan_fp'])

    # Prepare features (X) and target (y)
    features = np.array(list(lipo_data['morgan_fp'].values))
    targets = lipo_data['exp'].values 

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

    # Scale the target values
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel()

    # Train the MLP Regressor model
    mlp = MLPRegressor(hidden_layer_sizes=(n_estimators, max_depth), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train_scaled)

    # Predict on the test set
    y_pred_scaled = mlp.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE for Morgan fingerprints: {rmse:.4f}")

    # Get the conda environment name
    conda_env = os.getenv("CONDA_DEFAULT_ENV")

    # Prepare results for saving
    results = (
        f"Conda Environment: {conda_env}\n"
        f"Number of Estimators: {n_estimators}\n"
        f"Maximum Depth: {max_depth}\n"
        f"RMSE: {rmse:.4f}\n"
    )

    # Save results to a text file
    with open("results.txt", "w") as f:
        f.write(results)

    print("Results saved to results.txt")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Train a model on the Lipophilicity dataset using Morgan fingerprints.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators (neurons in the first hidden layer).")
    parser.add_argument("--max_depth", type=int, default=50, help="Maximum depth of the hidden layers.")
    
    args = parser.parse_args()
    
    # Call the main function with user-provided arguments
    main(args.n_estimators, args.max_depth)