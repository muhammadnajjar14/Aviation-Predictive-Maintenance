# Aviation-Predictive-Maintenance
A reproducible deep-learning pipeline for classifying pre-maintenance flights using NGAFID flight data

This repository contains the full workflow used in the paper “A Hybrid Generative and Anomaly-Based Framework for Predictive Maintenance on Imbalanced Aviation Time-Series Data.”
It provides everything needed to reproduce the results end-to-end: preprocessing, synthetic fault generation, classifier training, autoencoder analysis, and all uncertainty quantification experiments.

The focus is a difficult binary task on NGAFID data:
predicting whether a flight was healthy or likely to require maintenance soon.

The dataset is extremely imbalanced (≈ 8736 healthy vs. 30 faulty), so the repository includes the Bootstrap Crossover synthesis method used to balance the training distribution.


NGAFID Dataset

The National General Aviation Flight Information Database (NGAFID) provides non-simulated flight data from general aviation aircraft.
The dataset includes:

Multiple aircraft types

Flights from different seasons and environmental conditions

Associated maintenance logs

The faulty class consists of “pre-maintenance” flights — not catastrophic failures.
These signatures are subtle and often similar to healthy flights, which is why uncertainty analysis is included in the pipeline.

How to Reproduce All Results

Environment and Reproducibility

All experiments in this repository were developed and executed inside a dedicated Conda environment.


The environment is based on Python 3.10.18 and includes all packages required to run the full pipeline (data preparation, synthetic generation, classifier, autoencoder, and uncertainty analysis).

To make replication straightforward, three environment descriptions are provided:

---

1. `requirements.txt
Generated using:

```bash
pip freeze > requirements.txt

2. environment.yml (clean, portable)

Exported with:

conda env export --no-builds > environment.yml


This version omits platform-specific build numbers and is the most reliable way to recreate the environment on another machine:

conda env create -f environment.yml

OR

3. environment_with_builds.yml (exact snapshot)

This is a full pointer-accurate copy of the environment, exported using:

conda env export > environment_with_builds.yml


It captures the exact build variants present on the original Ubuntu workstation.
Use this file only if you require bit-for-bit reproducibility:

conda env create -f environment_with_builds.yml

Recommended Workflow for Replication

Clone this repository

git clone https://github.com/muhammadnajjar14/Aviation-Predictive-Maintenance.git
cd Aviation-Predictive-Maintenance


Create the environment (choose one of the three options):

environment.yml for portability

environment_with_builds.yml for exact reproduction

requirements.txt for pip setups

Launch the notebooks from inside the environment:

conda activate [ENVIRONMENT NAME]
jupyter notebook


Run the notebooks in the order:

01_Data_Preparation
02_Classifier_and_Generation
03_Anomaly_Autoencoder
04_Hybrid_UQ_Analysis

Each stage of the pipeline has a dedicated notebook.

1. Data Preparation

notebooks/01_Data_Preparation.ipynb

Load NGAFID flights

Standardize using healthy training flights only

Pad/truncate to fixed length

Export arrays for training

2. Synthetic Fault Generation + Classifier Training

notebooks/02_Classifier_and_Generation.ipynb

This notebook implements:

Residual extraction

Bootstrap Crossover synthesis

Balanced training set creation

Conv–BiLSTM classifier training

Evaluation metrics: AUROC, AUPRC, F1

. Autoencoder (Healthy-Only)

notebooks/03_Anomaly_Autoencoder.ipynb

Trains a convolutional autoencoder on healthy data only:

Calculate reconstruction error per flight

Determine anomaly threshold from validation distribution

Save AE scores for hybrid analysis

4. Uncertainty Quantification

notebooks/04_Hybrid_UQ_Analysis.ipynb

Contains:

MC-Dropout evaluation

Probability vs. uncertainty scatter

Hybrid probability vs. AE error analysis

All UQ figures from the paper

Key Results (Summary)

The Conv–BiLSTM achieves ≈ 85% recall on faulty flights.

High false-positive rate → consistent with subtle pre-maintenance signatures.

MC-Dropout uncertainties do not correlate with model errors.

The healthy-only autoencoder treats faulty flights as in-distribution.

Hybrid analysis shows no AE separation between false alarms and true faults.

These patterns suggest that ambiguity is driven by the limits of the sensor set, not model capacity.

License

This project is released under the MIT License.

Contribution

Issues and pull requests are welcome.
For major changes, please open an issue first to discuss the proposal.