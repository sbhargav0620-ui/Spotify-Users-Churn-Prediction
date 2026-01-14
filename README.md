# 🎧 Spotify User Churn Prediction

Predicting customer churn is a critical business objective for subscription-based services like Spotify. This project explores the machine learning pipeline required to identify at-risk users and provides actionable insights for retention strategies.

## 📌 Project Overview
The goal of this project was to move beyond simple accuracy and build a model capable of distinguishing between "loyal" and "at-risk" users in an imbalanced dataset. I evolved the model from basic Decision Trees to **Ensemble Methods** (Random Forest and XGBoost) to optimize for the business-critical metric: **Precision**.

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Matplotlib, Scikit-Learn, XGBoost
* **Techniques:** Bagging, Boosting, Probability Thresholding, Feature Engineering

---

## 📊 The Modeling Journey
I implemented a "Champion-Challenger" workflow to find the best-performing architecture:

1.  **Baseline (Decision Tree):** Highly interpretable but suffered from high variance and low recall.
2.  **Random Forest (Bagging):** Reduced variance by training multiple trees in parallel. 
3.  **XGBoost (Boosting):** Optimized the model through sequential learning. By focusing on the errors of previous trees, XGBoost provided the most "surgical" approach to identifying churners.

---

## 📈 Key Results & Metrics

| Model | Accuracy | Class 1 Precision | Class 1 Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **XGBoost (Default)** | 73% | 0.41 | 0.71 | 0.52 |
| **XGBoost (Threshold=0.7)** | **82%** | **0.60** | **0.37** | **0.45** |

### The "Accuracy Paradox"
A naive model that predicts "No Churn" for everyone would achieve **79.4% accuracy**. My final model achieves **82%**, representing a significant lift in identifying the actual 20% of users who are leaving.

---

## 🔍 Technical Deep Dive

### 1. The Precision-Recall Trade-off
In this project, I made a strategic decision to prioritize **Precision**. 
* **Strategy:** Manually adjusted the classification threshold to **0.7** (instead of 0.5).
* **Outcome:** Increased Precision to **0.60**. 
* **Business Impact:** Minimizes "False Positives," ensuring that expensive retention offers are only sent to users truly likely to leave.

### 2. Feature Importance
Using XGBoost’s built-in feature importance, I identified that **Support Tickets** and **Days Since Last Login** are the primary drivers of churn.

---

## 🚀 Lessons Learned & Future Work
* **Data Constraints:** The model currently uses static averages. Future iterations will involve **Time-Series Analysis** (e.g., "Drop in usage over the last 7 days vs. 30 days") to capture behavioral momentum.
* **Class Imbalance:** Successfully mitigated imbalance using `scale_pos_weight`, but further performance could be unlocked using **SMOTE** or targeted data collection.

---

## ⚙️ How to Run
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/your-username/spotify-churn-prediction.git](https://github.com/your-username/spotify-churn-prediction.git)
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    Open `notebooks/churn_modeling.ipynb` to see the full analysis.
