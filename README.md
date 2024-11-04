# Lipophilicity Fingerprint Model

This project trains a MLPregression model to predict lipophilicity values using Morgan Fingerprints. It uses a machine learning model with hyperparameters specified by the user.

## Installation

Clone this repository and set up the environment:

```bash
git clone https://github.com/your-username/LipophilicityFingerprintModel.git
cd LipophilicityFingerprintModel
conda env create -f environment.yml
conda activate lipophilicity_env

# Running the model, can specify the hyperparameters differently:
```bash 
python lipophilicity_model/train_model.py --n_estimators 100 --max_depth 50