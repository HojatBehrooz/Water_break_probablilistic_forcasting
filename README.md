# Water_break_probablilistic_forcasting
A Deep Learning Auto-Regressive Forecasting Model for Probabilistic Water Pipe Break Prediction
# Water Pipe Break Prediction Using Deep Learning

## Abstract
Accurately predicting the likelihood of water pipe breaks is pivotal for proactive maintenance, cost-effective emergency repairs, and mitigating service disruptions. However, crafting a dependable predictive model for water pipeline breaks is formidable. The challenges stem from the sporadic and infrequent occurrences of breaks, irregular intervals between failures, intricate temporal dependencies among pipes with diverse attributes, and the unbalanced distribution of historical data. Although a considerable number of studies in recent years have developed forecasting models using classic statistical techniques, machine learning solutions, and deep learning methods, state-of-the-art models have yet to achieve the level of predictive power needed to help utilities transform their practices for risk-based proactive maintenance. This study addresses this need by developing and empirically examining the performance of a novel deep learning-based auto-regressive forecasting model for probabilistic water pipe break prediction.

Notably, the proposed probabilistic forecasting method integrates a multivariate/multidimensional auto-regressive model with a Recurrent Neural Network (RNN) in the form of a Long Short-Term Memory (LSTM) model to capture complex and irregular temporal patterns, characterizing dependencies and interrelationships among the time series of pipeline attributes over time and transforming the apprehended patterns into a probabilistic pipe failure prediction through a distribution-based mechanism. The proposed method was implemented to predict the likelihood of water pipe breaks in Calgary, Canada, using historical data from 1956 to 2022. The outcomes indicated that the proposed model maintains a very strong predictive power, achieving an Area Under the Curve (AUC) measure of 99.98% in predicting breaks in 2023. The outcomes of this study will help decision-makers plan risk-based maintenance operations that prevent service disruptions and safeguard public health.

**Keywords**: Water Pipe Break; Probabilistic Prediction; Deep Learning; Auto-Regressive; LSTM.

## Files Description
This repository contains the following Python scripts:

### 1. `Data_preparation.py`
- **Description**: This script prepares the dataset by reading input data files, composing, and preprocessing them into a format suitable for the main processing script.
- **Functionality**:
  - Reads raw data files from the `input` directory.
  - Cleans and preprocesses the data, handling missing values and data transformations.
  - Outputs a processed dataset ready for scaling and training in `Main.py`.

### 2. `Main.py`
- **Description**: The main script that handles the scaling of data, training of the LSTM model, and making predictions.
- **Functionality**:
  - Loads the preprocessed data and applies scaling.
  - Implements the LSTM-based deep learning model for training on the historical dataset.
  - Outputs predictive results for the given time period.
  - Evaluates the model performance, providing metrics such as AUC.

### 3. `SVC implementation as a reference to the paper by Robles-Velasco (2020).py`
- **Description**: This script re-implements the Support Vector Classification (SVC) method described in Robles-Velasco (2020), applying it to the same dataset used in the LSTM model to compare AUC accuracy.
- **Functionality**:
  - Loads the dataset and preprocesses it for the SVC model.
  - Trains the SVC model using the radial basis kernel function as described in the reference paper.
  - Outputs performance metrics, including AUC, for comparison with the proposed LSTM model.

## Imported Packages
The following key Python libraries and packages are used in this project:

- **GluonTS 0.15.1**:

## Additional Notes
- **Dependencies**: Ensure that Python libraries such as `pandas`, `scikit-learn`, `tensorflow`, `gluonts`, and `matplotlib` are installed to run the scripts.
- **Data Directory**: The `input` directory should contain all relevant datasets required by `Data_preparation.py` for preprocessing.
- **Usage**: Follow the order of execution—first run `Data_preparation.py` to prepare the data, then `Main.py` for training and prediction, and finally, the SVC script for benchmarking
