# Employee Insurance Enrollment Prediction

This project builds a machine learning model to predict employee insurance enrollment based on demographic and employment data.

## Project Overview

The goal is to accurately predict whether an employee will enroll in an insurance plan based on various features. The model achieves this by:

1. Preprocessing and analyzing employee data
2. Applying feature engineering techniques
3. Training and evaluating multiple machine learning models
4. Selecting the best model for deployment
5. Optioally deploying the model using FastAPI
6. Tracking experiments using MLflow

## Installation and Setup

### Prerequisites
- [Anaconda](https://www.anaconda.com/download/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

### Setting up the Environment

1. Clone the repository:
```bash
git clone https://github.com/srajanseth84/uniblox-ml-assisnment
cd uniblox-ml-assisnment
```

2. Create a Conda environment and install dependencies:
```bash
conda env create -f environment.yml
```
3. Activate the environment:
```bash
conda activate insurance-pred
```
### Directory Structure
```
├── data/
│   └── employee_data.csv
├── models/            # Saved model files
├── preprocessor/      # Saved preprocessor files
├── mlruns/            # MLflow runs
├── images/            # Some images used in the report
├── server/            # FastAPI server files
│   ├── main.py        # FastAPI server entry point
│   ... 
├── environment.yml    # Conda environment file
├── notebook.ipynb     # Jupyter notebook with analysis and model training
├── report.md          # Detailed report
└── README.md          # This file
```

## Running the Server

1. Change to server directory:
```bash
cd server
```
2. Run the server:
```bash
python main.py
```
3. Access the API documentation at `http://localhost:8000/docs` to explore the available endpoints.

## MLflow Tracking


1. If you are in server directory, change to the root directory:
```bash
cd ..
```

1. To view the MLflow tracking UI, run the following command:
```bash
mlflow ui --backend-store-uri file://$(pwd)/mlruns      
```
Then, open your browser and navigate to `http://localhost:5000` to view the experiment results.

## Author

Srajan Seth
