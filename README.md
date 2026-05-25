# 🏃 Human Activity Recognition (HAR) Pipeline & Dashboard

An end-to-end Machine Learning and Deep Learning pipeline that classifies human activities using smartphone inertial sensor data. This project explores traditional ML techniques, advanced sequence modeling with LSTMs, and culminates in a real-time interactive web dashboard built with Streamlit.

---

## 📖 Project Overview

This repository demonstrates a complete data science and machine learning lifecycle using the **UCI HAR Dataset**. The goal is to accurately predict six human activities (*Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying*) based on high-dimensional sensor data (accelerometer and gyroscope) captured from a smartphone.

The project is divided into three major phases:
1. **Classical Machine Learning:** Feature engineering, PCA, and baseline classification models.
2. **Deep Learning:** Time-series sequence modeling using Recurrent Neural Networks (LSTM) on raw 3D inertial signals.
3. **Interactive Deployment:** A live Streamlit dashboard to simulate and predict activities in real-time.

---

## 📊 The Dataset

**[UCI Machine Learning Repository: Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)**

* **Source:** 30 volunteers performing activities while wearing a smartphone (Samsung Galaxy S II) on their waist.
* **Signals:** 3-axial linear acceleration and 3-axial angular velocity captured at a constant rate of 50Hz.
* **Preprocessing applied:** Noise filters, sampled in fixed-width sliding windows of 2.56 sec and 50% overlap (128 readings/window).

*(Note: The provided scripts automatically download and extract this dataset via `wget` and `unzip`.)*

---

## 🧠 Phase 1: Classical Machine Learning Pipeline

In the first phase, we utilize the pre-extracted, hand-crafted features provided by the UCI dataset (561 features).

* **Data Preprocessing:** Standardizes the feature set using `StandardScaler`.
* **Dimensionality Reduction (PCA):** Applies Principal Component Analysis to reduce the 561 features down to the components that explain 95% of the variance, optimizing training time and reducing noise.
* **Model Training & Comparison:**
  * **Logistic Regression:** Trained as a baseline multinomial classifier.
  * **Random Forest:** Trained with 200 estimators.
* **Model Evaluation:** Both models are evaluated using Accuracy, Precision, and Log Loss.
* **Complexity Analysis:** Generates **Validation Curves** to plot Model Complexity (C for Logistic Regression, max_depth for Random Forest) against Log Loss to identify and prevent overfitting and underfitting.

---

## 🚀 Phase 2: Deep Learning (LSTM) on Raw Time-Series Data

Rather than relying on the 561 hand-crafted features, this phase builds a deep learning model that learns directly from the raw sequential data.

* **Data Loading:** Loads the raw 3D inertial signals (`body_acc`, `body_gyro`, `total_acc` across X, Y, Z axes).
* **Shape:** Transforms data into a 3D tensor: `(samples, 128 timesteps, 9 features)`.
* **Standardization:** Reshapes the 3D tensor to 2D for `StandardScaler`, then reshapes back to sequence format.
* **LSTM Architecture:**
  * LSTM Layer (128 units, return sequences) + Dropout (0.5)
  * LSTM Layer (64 units) + Dropout (0.5)
  * Dense Output Layer (6 units, Softmax activation)
* **Optimization:** Uses the `Adam` optimizer (learning rate = 0.0005) with `Categorical Crossentropy`.
* **Callbacks:** Implements `EarlyStopping` (restores best weights) and `ReduceLROnPlateau` to dynamically adjust the learning rate during training.

---

## 🌐 Phase 3: Streamlit Interactive Dashboard

The project features a deployed web application (`app.py`) built with **Streamlit** that utilizes the trained LSTM model (`activity_model.keras`) to make predictions. 

### Dashboard Input Modes:
1. **Manual Sliders:** Allows users to manually adjust the 9 sensor features (X, Y, Z for Body Accel, Gyro, and Total Accel) via UI sliders to simulate a steady physical state and see instant predictions.
2. **Upload CSV:** Batch processing. Users can upload a CSV of raw sensor data (at least 128 rows x 9 columns) to simulate a recorded activity sequence.
3. **Simulated Stream:** Generates real-time, active simulated sensor noise to demonstrate how the model processes continuous live data, accompanied by a dynamic line chart and confidence metrics.

---

## 🛠️ How to Run

### 1. Installation
Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install numpy pandas scikit-learn matplotlib tensorflow streamlit pyngrok
```

### 2. Train the Models & Run the Notebook
Execute the main Python notebook/script to run the pipeline. This step will automatically:
1. Download and extract the dataset.
2. Run the Classical ML comparisons and plot the complexity curves.
3. Train the LSTM model and save it locally as `activity_model.keras`.

### 3. Launch the Web Dashboard
Once the `activity_model.keras` file is generated, you can launch the Streamlit app.

**Running Locally:**
```bash
streamlit run app.py
```

**Running in Google Colab:**
If you are executing this within Google Colab, the code is already configured to use `pyngrok` to tunnel the local Streamlit server to a public, accessible URL. 
* *Ensure you replace the placeholder `NGROK_AUTH_TOKEN` in the script with your actual token from the Ngrok dashboard.*
* The script will output a live URL where you can interact with the app.

---

## 📂 Repository File Structure
* `HAR_Pipeline.ipynb` — The main script containing data processing, ML, and DL model training.
* `app.py` — The Streamlit application source code.
* `activity_model.keras` — The compiled, trained Deep Learning model (generated after training).
* `UCI HAR Dataset/` — Directory containing the downloaded dataset (generated automatically).
