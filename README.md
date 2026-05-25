
🏃 Human Activity Recognition (HAR) Pipeline & Dashboard
An end-to-end Machine Learning and Deep Learning pipeline that classifies human activities using smartphone inertial sensor data. The project compares traditional ML models with an LSTM neural network and features a live interactive web dashboard built with Streamlit to predict activities in real-time.

Overview
This project utilizes the UCI HAR Dataset to predict human activities: Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying. It processes high-dimensional sensor data (accelerometer and gyroscope) to train predictive models.

The repository demonstrates a full data science lifecycle:

Automated Data Ingestion: Downloading and extracting the UCI HAR dataset.

Classical Machine Learning: Feature scaling, PCA dimensionality reduction, and training Logistic Regression & Random Forest classifiers.

Deep Learning: Building and training an LSTM network on raw sequential time-series data.

Interactive Deployment: A Streamlit web dashboard exposed via Pyngrok to test predictions interactively.

Features
Baseline ML Models: Evaluates Logistic Regression and Random Forest using accuracy, precision, and log loss.

Complexity Analysis: Generates validation curves to analyze model complexity vs. loss (preventing overfitting/underfitting).

Time-Series Deep Learning: Implements a robust LSTM model with Dropout, EarlyStopping, and learning rate scheduling to process 128-timestep sequences.

Streamlit Web App: A deployed UI with three distinct input modes:

Manual Sliders: Adjust 9 sensor features manually to simulate steady states.

Upload CSV: Batch process raw sensor data to predict activities.

Simulated Stream: Real-time dashboard generating mock sensor signals and live predictions.

Technologies Used
Language: Python

Data Processing: pandas, numpy, scikit-learn

Deep Learning: tensorflow, keras

Visualization: matplotlib

Web App Deployment: streamlit, pyngrok

Dataset
UCI Machine Learning Repository: Human Activity Recognition Using Smartphones

The dataset contains 3-axial linear acceleration and 3-axial angular velocity captured at a constant rate of 50Hz using a Samsung Galaxy S II.

The code automatically downloads and unzips this dataset during execution.

How to Run
1. Prerequisites
Ensure you have Python 3.7+ installed. Install the required dependencies:

Bash
pip install numpy pandas scikit-learn matplotlib tensorflow streamlit pyngrok
2. Train the Models
Run the Jupyter Notebook (or Python script). The script will:

Download the dataset.

Train the Scikit-learn models (and output their performance graphs).

Train the LSTM model.

Save the trained deep learning model as activity_model.keras in your root directory.

3. Run the Streamlit App
If you are running this locally (not in Colab), you can start the Streamlit dashboard directly from your terminal:

Bash
streamlit run app.py
(Note: If you are running this in Google Colab, the script is already configured to use pyngrok to tunnel the local Streamlit server to a public URL. Just make sure to replace the NGROK_AUTH_TOKEN in the script with your own token from the Ngrok dashboard).

Model Performance
Classical ML: Random Forest and Logistic Regression provide strong baselines after PCA dimensionality reduction (capturing 95% of variance).

LSTM Neural Network: By bypassing manual feature extraction and learning directly from the raw 3D inertial signals (samples, 128 timesteps, 9 features), the LSTM captures the temporal dependencies of the physical movements.

 Project Structure (Generated)
app.py — The Streamlit application script.

activity_model.keras — The compiled and trained LSTM model.

UCI HAR Dataset/ — The extracted dataset directory containing train/test splits and inertial signals.
