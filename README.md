Aviation-Predictive-Maintenance

A reproducible deep-learning pipeline for identifying pre-maintenance flights using NGAFID multivariate time-series data.

This repository accompanies the paper:

“A Hybrid Generative and Anomaly-Based Framework for Predictive Maintenance on Imbalanced Aviation Time-Series Data.”

It contains the full workflow needed to reproduce all results end-to-end:
data preparation, residual-based synthetic fault generation, classifier training, anomaly autoencoder modeling, and the complete uncertainty-quantification analysis.

NGAFID Dataset Overview

The National General Aviation Flight Information Database (NGAFID) provides non-simulated flight data from general-aviation aircraft.

This project focuses on a binary task:

Healthy Flights

Pre-Maintenance Flights (i.e., flights that later triggered a maintenance event)

The challenge is the extreme class imbalance:

~8,736 healthy

~30 faulty

Pre-maintenance signatures tend to be subtle and often resemble normal operation, which motivates the generative balancing and uncertainty-analysis methods implemented here.

Reproducing the Pipeline
Environment and Reproducibility

All experiments were executed inside a dedicated Conda environment running:

Python 3.10.18

To make replication straightforward, three environment files are included:

1. requirements.txt (pip-based)

Generated using:

pip freeze > requirements.txt


Install with:

pip install -r requirements.txt

2. environment.yml (portable Conda environment)

Exported with:

conda env export --no-builds > environment.yml


Recreate with:

conda env create -f environment.yml
conda activate rapids_muhammad

3. environment_with_builds.yml (exact snapshot)

Exported with:

conda env export > environment_with_builds.yml


Recreate with:

conda env create -f environment_with_builds.yml
conda activate rapids_muhammad


Use this version only if exact replication of the original Ubuntu system matters.

Recommended Workflow

Clone the repository:

git clone https://github.com/muhammadnajjar14/Aviation-Predictive-Maintenance.git
cd Aviation-Predictive-Maintenance


Create and activate the environment (choose one of the three options above), then launch Jupyter:

jupyter notebook

Pipeline Structure

Run the notebooks in this order:

1. Data Preparation

notebooks/01_Data_Preparation.ipynb

Load NGAFID flight data

Standardize using healthy training flights only

Pad/truncate sequences to length 2048

Save structured arrays for training and evaluation

2. Synthetic Generation + Classifier Training

notebooks/02_Classifier_and_Generation.ipynb

Implements:

Trend–residual decomposition

Bootstrap Crossover synthesis to balance the data

Dataset assembly for 1:1 class ratio

Conv–BiLSTM classifier training

Evaluation metrics: AUROC, AUPRC, Recall, F1

3. Healthy-Only Anomaly Autoencoder

notebooks/03_Anomaly_Autoencoder.ipynb

Convolutional autoencoder trained only on healthy flights

Compute reconstruction error per sample

Determine anomaly threshold from healthy validation distribution

Export anomaly scores for hybrid analysis

4. Uncertainty Quantification & Hybrid Analysis

notebooks/04_Hybrid_UQ_Analysis.ipynb

Contains:

MC-Dropout probability vs. uncertainty evaluation

Scatter plots for correct vs. incorrect predictions

Reconstruction-error analysis from the autoencoder

Hybrid UQ plot combining classifier probability & AE anomaly score

All figures reported in the paper

Key Results

Conv–BiLSTM classifier achieves ≈ 85% recall on faulty flights.

False positives remain common due to the subtle nature of pre-maintenance signatures.

MC-Dropout uncertainty does not correlate with model errors.

The healthy-only autoencoder treats most faulty flights as in-distribution.

Hybrid probability × reconstruction-error analysis shows no separation between true faults and false alarms.

These behaviors point to intrinsic ambiguity in the available sensor channels, not model limitations.

License

Released under the MIT License.

Contributing

Issues and pull requests are welcome.
For major changes, please open an issue first so we can discuss them.
