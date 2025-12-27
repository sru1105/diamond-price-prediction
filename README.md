# Diamond Price Prediction

This project analyzes the impact of physical diamond attributes (the 4Cs and dimensions) on market price and implements predictive modeling using high-performance regression algorithms.

## Overview
Diamonds are reliable investment assets due to their low market volatility. This project determines how features like Carat, Cut, Clarity, Color, and physical dimensions (x, y, z, depth, table) affect market valuation.

## Dataset
[cite_start]The project utilizes a dataset of **53,943 observations** with 11 features[cite: 1, 4]:
- [cite_start]**Physical Attributes:** Carat, x, y, z, depth, table[cite: 4].
- [cite_start]**Categorical Features:** Cut (Fair to Ideal), Color (J to D), and Clarity (I1 to IF)[cite: 4].
- [cite_start]**Target Variable:** Price (USD)[cite: 4].

## Methodology
The implementation follows a comprehensive data science pipeline:
1. **Exploratory Data Analysis (EDA):** Visualizing distributions and feature correlations using Seaborn and Matplotlib.
2. **Feature Selection & Optimization:**
   - **Correlation Analysis:** Used to identify and mitigate multicollinearity between features.
   - **Dimensionality Reduction:** Applied **PCA** and **Recursive Feature Elimination** to obtain the best subset of features.
   - **Statistical Testing:** Conducted **Chi-square tests** to validate feature significance.
3. **Modeling:** Implemented various regression algorithms, including Linear Regression, Random Forest, and CatBoost.

## Key Results
- **CatBoost Regression** achieved the highest accuracy of 0.9872, outperforming other models.
- **Random Forest Regression** followed closely with an **R² score of 0.9800**
- Other models tested included Linear Regression, Support Vector Regression, Decision Tree, and XGBoost

## Technologies Used
- **Language:** Python
- **ML Frameworks:** CatBoost, Scikit-learn
- **Data Libraries:** Pandas, NumPy, PCA
- **Visualization:** Matplotlib, Seaborn