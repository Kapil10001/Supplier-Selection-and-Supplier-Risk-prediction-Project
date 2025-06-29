# Supplier-Selection-and-Supplier-Risk-prediction-Project 
# Supplier Risk Prediction and Optimization using Logistic Regression & MILP

# Project Overview

This project addresses a real-world supply chain problem in a mobile manufacturing company, where multiple suppliers are available for sourcing critical components. The goal is to **predict the risk of supplier default** using machine learning and then **select the optimal set of suppliers** for multiple products using **Mixed Integer Linear Programming (MILP)** to minimize cost and risk.



# Tools and Technologies
- Python
- Pandas, Scikit-learn – for data analysis and logistic regression
- Google OR-Tools – for optimization modeling
- Matplotlib/Seaborn – for data visualization



# Problem Statement

Given:
- A dataset of 5 suppliers with features like `Cost`, `Rating`, `Reliability`, etc.
- Demand requirements for 8 different products
- Historical supplier performance

Objective:  
1. Predict the probability of Default of  each supplier using logistic regression.  
2. Use these probabilities in an optimization model to decide:
   - Which suppliers to choose?
   - How much to procure from each?
   - While minimizing the "Total cost" (procurement + risk + fixed cost).



# Machine Learning Model

- Model Used: Logistic Regression  
- Target Variable: Supplier Default (0: No, 1: Yes)  
- Features: Cost, Rating, Reliability, Past Performance, etc.  
- Accuracy Achieved: `82%`

The output (`prob_default`) is used as a penalty in the optimization objective.


# Optimization Model

- Decision Variables:
  - `x[i][j]`: Number of units of product `j` ordered from supplier `i`
  - `y[i]`: Binary variable = 1 if supplier `i` is selected

- Objective Function:
  Minimize:  
  `Total_Cost = ∑(x[i][j] * unit_cost[i][j]) + ∑(y[i] * fixed_cost[i]) + ∑(prob_default[i] * risk_penalty)`

- Constraints:
  - Meet product demand
  - Respect supplier capacities
  - Limit number of suppliers 



# Results

- Final supplier selection list for each product
- Total cost minimized while factoring in risk
- Demonstrated cost saving and risk mitigation versus random/manual allocation



# Validation Approach

- Compared optimized vs. manual selection strategies
- Verified demand satisfaction and cost breakdown
- Evaluated supplier performance alignment with risk prediction



##
This project tackles a real-world supply‑chain challenge by combining machine learning and mathematical optimization to make smarter, risk‑aware purchasing decisions. First, I built a logistic regression model that analyzes historical supplier data (cost, quality, delivery performance, financial health, etc.) to predict each supplier’s probability of default. Those risk scores then feed into a Mixed‑Integer Linear Programming model (implemented with Google OR‑Tools), which decides both **which** suppliers to engage (binary selection) and **how much** of each product to order from them (continuous quantities). The objective function minimizes the total cost—including discounted purchase prices, fixed administration fees, and risk penalties—while ensuring all product demands are met and no supplier is over‑relied upon. The result is a fully automated, end‑to‑end solution that not only cuts procurement costs but also builds resilience by balancing price advantages with supplier reliability.







