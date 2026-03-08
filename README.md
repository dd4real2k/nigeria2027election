# 🗳️ Nigeria 2027 Presidential Election Prediction
### Data Science Portfolio Project

---

## 📌 Problem Statement

Nigeria has conducted democratic presidential elections since 1999, yet voting outcomes are influenced by many complex factors including economic conditions, demographics, historical voting patterns and regional dynamics.

Understanding these patterns is valuable for political scientists, economists, and data scientists studying democratic processes.

This project explores whether historical election data, economic indicators, and demographic variables can be used to build machine learning models capable of identifying patterns and predicting potential election trends.

The goal is not to determine election outcomes, but to demonstrate how data science techniques can be applied to large-scale political and socio-economic datasets.

---

## 📌 Project Overview

This project applies machine learning, statistical modelling, and data visualisation
to predict the outcome of Nigeria's 2027 Presidential Election (scheduled January 16, 2027).

It demonstrates a full data science workflow — from data collection to deployment —
making it ideal as a portfolio project for a data analyst role.

---

## 🗂️ Project Structure

```
nigeria2027/
├── data/
│   ├── raw/
│   │   ├── nigeria_election_results_1999_2023.csv
│   │   ├── nigeria_economic_indicators.csv
│   │   ├── nigeria_state_demographics.csv
│   │   └── nigeria_polls_2024_2025.csv
│   └── processed/
├── notebooks/
│   └── nigeria_2027_prediction.py
├── app/
│   └── dashboard.py
├── models/
├── visuals/
└── README.md

```

## 📦 Dataset Overview
This project combines multiple public datasets covering elections, economics, demographics, and polling data.

| Dataset | Source | Description |
|---|---|---|
| Election Results 1999–2023 | INEC / GitHub: nigeria2/data-ng-election-2023 | State-level votes per candidate per election |
| Economic Indicators | NBS, World Bank, AfDB | GDP growth, inflation, unemployment, oil price |
| State Demographics | NBS, Afrobarometer | Religion, ethnicity, literacy, poverty, security |
| Polling Data | NOIPolls, Afrobarometer, SBM Intelligence | 2024–2025 approval ratings |

### 🔗 Real Data Sources
- **INEC Results**: https://www.inecelectionresults.ng
- **GitHub CSV**: https://github.com/nigeria2/data-ng-election-2023
- **NBS Statistics**: https://nigerianstat.gov.ng/elibrary
- **World Bank Nigeria**: https://data.worldbank.org/country/nigeria
- **Afrobarometer**: https://www.afrobarometer.org
- **NOIPolls**: https://noipolls.com
- **ACLED Security Data**: https://acleddata.com (free registration)

---

## ⚙️ Tech Stack

Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib / Seaborn
Plotly
Streamlit
Jupyter Notebook

---

## 🤖 Machine Learning Pipeline

The modelling workflow follows a structured machine learning pipeline:

1. Data Cleaning
 - Missing value handling
 - Feature normalization
 - Encoding categorical variables

2. Feature Engineering
 - Key engineered variables include:
 - Economic stress index
 - Incumbent approval proxy
 - Regional voting blocs
 - Religious composition indicators
 - Turnout deviation metrics
 - Security risk scores
 - State electoral weight

3. Model Training
Several machine learning models were trained and compared:

| Model               | Purpose                     |
| ------------------- | --------------------------- |
| Logistic Regression | Baseline model              |
| Random Forest       | Primary model               |
| XGBoost             | High-performance challenger |

4. Model Evaluation
Models were evaluated using:
 - Cross-validation
 - Accuracy
 - Feature importance analysis
 - Scenario testing

5. Simulation
Monte Carlo simulation (10,000 election trials) was used to estimate probabilistic outcomes under different political scenarios.

## 🗺️ Project Architecture

Historical Election Data (1999–2023)
           ↓
Data Cleaning & Processing
           ↓
Feature Engineering (21 variables)
           ↓
Machine Learning Models
(LR + Random Forest + XGBoost)
           ↓
State-Level Predictions
           ↓
Monte Carlo Simulation (10,000 runs)
           ↓
Scenario Analysis
           ↓
Interactive Dashboard


## 📊 Key Findings

1. **Fragmented opposition** is APC's biggest advantage — as seen in 2023 (36.6% win)
2. **Economic conditions** (inflation especially) are the strongest predictor of incumbent loss
3. **North-West bloc** (Kano, Katsina, Kaduna, etc.) is decisive — ~8M+ votes
4. **United opposition** (ADC scenario) flips the predicted outcome
5. **2027 base case**: APC has ~42–48% win probability under fragmented opposition


## 🚀 How to Run

### 1. Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly streamlit scipy
```

### 2. Launch the interactive dashboard
```bash
cd app
streamlit run dashboard.py
```

## 📊 Interactive Dashboard

The project includes a Streamlit dashboard allowing users to explore:
- Historical voting trends
- Economic indicators over time
- State-level electoral patterns
- Model predictions and simulations

Example visualisations include:
- Voting trend charts
- Feature importance graphs
- Monte Carlo probability distributions
- Scenario comparison charts

---

## 🔮 Future Improvements

Potential extensions to the project include:
- State-level election prediction maps
- Voter turnout prediction models
- Time-series forecasting of political sentiment
- Integration of real-time polling data
- Geospatial visualization of election outcomes


## ⚠️ Limitations & Disclaimer
- This project is for educational and portfolio purposes  only
- Election prediction involves high uncertainty
- Polling data used for 2024–2025 is illustrative
- The model does not account for last-minute political developments, electoral disputes, or candidate withdrawals

---

## 👤 Author

**Daniel Diala**
LinkedIn: https://www.linkedin.com/in/danieldiala/
GitHub: https://github.com/dd4real2k

---

*Election date: January 16, 2027 | Model last updated: March 2026*
