# ⚽ Prediction of Football Transfer Values (2025/26 Season)

## 📌 Overview

This project aims to predict football players' market transfer values for the **2025/26 season** using a data-driven approach based on the **Top 5 European Leagues**:

* Premier League (EPL)
* La Liga
* Serie A
* Bundesliga
* Ligue 1

By leveraging player statistics and machine learning models, this project provides insights to support **data-driven decision-making in football transfers**.

---

## 🎯 Objectives

* Analyze key factors affecting player market value
* Build a comprehensive dataset from multiple sources
* Apply machine learning models for value prediction
* Compare model performance and identify the best approach

---

## 📊 Dataset

The dataset is collected from:

* Transfermarkt (market value, player profile)
* WhoScored (performance statistics)

### 🔹 Key Information

* ~2000+ players
* ~70 features
* Includes:

  * Basic info: age, height, position, club
  * Attacking stats: goals, shots, assists
  * Defensive stats: tackles, interceptions
  * Passing stats: passes, key passes

---

## ⚙️ Data Pipeline

### 1. Data Collection

* Web scraping using **Selenium**
* Multi-source data integration

### 2. Data Preprocessing

* Handle missing values
* Normalize formats (e.g., height, age)
* Log transformation of target variable

### 3. Feature Engineering

* Contract → `Month_Left`
* Nation → `Continent`
* Position → `Position_Group`
* Cards → `Card`

### 4. Feature Selection

* Correlation matrix
* Variance Inflation Factor (VIF)
* Final selected features: **48**

### 5. Outlier Handling

* IQR (preferred over Winsorizing)

---

## 🤖 Models Used

The following regression models were implemented:

* Ridge Regression
* Support Vector Regression (SVR)
* Random Forest
* XGBoost
* LightGBM

---

## 📈 Evaluation Metrics

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)
* R² Score

---

## 🏆 Results

* Ensemble models (XGBoost, LightGBM) achieved the best performance
* Data preprocessing significantly improved model accuracy
* Log transformation reduced skewness and stabilized training


## 🔧 Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* XGBoost, LightGBM
* Selenium (Web Scraping)
* Matplotlib / Seaborn

---

## ⚠️ Limitations

* Market values are estimated (not real transfer fees)
* Hard to predict superstar players due to external factors
* Data collected from public sources

---

## 🔮 Future Work

* Apply Deep Learning models
* Integrate real-time transfer data
* Use NLP for news sentiment analysis
* Combine Reinforcement Learning for transfer strategy

---

## 👨‍💻 Authors

* Ha Xuan Hoang
* Nguyen Huynh Minh Phu
* Nguyen Gia Tuan Anh
* Tran Quoc Khanh


