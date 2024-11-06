# Lipophilicity Fingerprint Model

This project trains a MLPregression model to predict lipophilicity values using Morgan Fingerprints. It uses a machine learning model with hyperparameters specified by the user.

## Installation

Clone this repository and set up the environment:

```bash
git clone https://github.com/your-username/LipophilicityFingerprintModel.git
cd LipophilicityFingerprintModel
conda env create -f environment.yml
conda activate lipophilicity_fingerprint_model
```
## Usage Code (Can vary n_estimators and max_depth values, results.txt used 100 and 50 respectively)

``` bash
python lipophilicity_model/train_model.py --n_estimators 100 --max_depth 50