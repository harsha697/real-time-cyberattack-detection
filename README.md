# Real-Time Cyberattack Detection Using Machine Learning and Zeek

## 📌 Project Overview

This project implements a **real-time network intrusion detection system (NIDS)** by combining **machine learning models** with **Zeek network monitoring logs**. The system is trained on the **UNSW-NB15 dataset** and is capable of detecting malicious network traffic in near real-time.

The core idea is to bridge **traditional network monitoring (Zeek)** with **modern ML-based anomaly detection**, achieving higher detection capability for complex and evolving cyberattacks.

---

## 🎯 Objectives

* Detect cyberattacks from network traffic with high accuracy
* Reduce false negatives while maintaining acceptable false positives
* Support **real-time inference** using Zeek logs
* Compare and combine multiple ML models
* Build a deployable, modular, and extensible IDS pipeline

---

## 🧠 Models Used

The project explores multiple machine learning models:

### 1. Random Forest Classifier

* Strong baseline model
* Handles non-linear relationships well
* Robust to noise and overfitting

### 2. XGBoost Classifier

* Gradient-boosted decision trees
* Performs well on structured/tabular data
* Handles class imbalance effectively

### 3. Ensemble Model (Soft Voting)

* Combines **Random Forest + XGBoost**
* Uses **probability-based (soft) voting**
* Improves generalization and recall

---

## 📊 Dataset

* **Dataset:** UNSW-NB15
* **Type:** Network intrusion dataset
* **Classes:**

  * 0 → Normal traffic
  * 1 → Attack traffic

### Preprocessing Steps

* Label encoding of categorical features
* Feature scaling (StandardScaler)
* Feature selection
* Class imbalance handling

---

## 🧪 Model Performance (Ensemble)

| Metric             | Value    |
| ------------------ | -------- |
| Accuracy           | **~72%** |
| Precision (Attack) | ~72%     |
| Recall (Attack)    | **~63%** |
| F1-score (Attack)  | ~77%     |

### Confusion Matrix

* True Positives (Attacks detected): **37,625**
* False Negatives (Missed attacks): **7,707**

📌 **High recall is prioritized**, making the system suitable for security environments where missing attacks is costly.

---

## ⚙️ Threshold Optimization

Instead of using a fixed 0.5 decision threshold:

* Model probabilities are evaluated across multiple thresholds
* Optimal threshold is selected using **F1-score maximization**
* Final threshold used: **0.10**

This improves attack detection sensitivity in real-world scenarios.

---

## 🛰️ Real-Time Detection with Zeek

### Workflow

1. Zeek captures live network traffic
2. Logs are parsed and transformed into feature vectors
3. Pre-trained ensemble model performs inference
4. Detected attacks are logged and alerted

### Script

* `zeek_realtime_detector_ubuntu.py`

This enables near real-time cyberattack detection on live networks.

---

## 📁 Project Structure

```
capstone_project/
│── preprocess_unsw.py
│── feature_selection.py
│── train_model.py
│── train_xgboost.py
│── train_ensemble.py
│── evaluate_model.py
│── evaluate_threshold.py
│── evaluate_xgboost_threshold.py
│── zeek_realtime_detector_ubuntu.py
│── .gitignore
│── README.md
```

---

## 🛠️ Technologies Used

* **Python 3.10**
* **Scikit-learn**
* **XGBoost**
* **Pandas / NumPy**
* **Zeek IDS**
* **Joblib**

---

## 🚀 How to Run

### 1. Create virtual environment

```bash
python3 -m venv cyberenv
source cyberenv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Preprocess data

```bash
python3 preprocess_unsw.py
```

### 4. Train ensemble model

```bash
python3 train_ensemble.py
```

### 5. Run real-time detection

```bash
python3 zeek_realtime_detector_ubuntu.py
```

---

## 📌 Key Takeaways

* Ensemble learning significantly improves intrusion detection
* Threshold tuning is critical for security-focused ML systems
* Zeek + ML provides a powerful hybrid IDS
* Recall is more important than raw accuracy in cybersecurity

---

## 🔮 Future Enhancements

* Multi-class attack classification
* Deep learning models (LSTM / CNN)
* Online learning for evolving threats
* Web dashboard for alerts and visualization

---

## 👨‍💻 Author

SIVAKOTI HARSHAVARDHAN

---

## ⭐ Acknowledgements

* UNSW Canberra Cybersecurity Research Group
* Zeek (formerly Bro) Network Security Monitor
* Open-source ML community

---

⭐ *If you find this project useful, consider starring the repository!*
