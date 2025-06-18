# 💡 Life Decision Simulator  
AI-Powered Guidance for Major Life Choices  

**By Saadat I.**  
🌐 https://github.com/saadatibk/life-decision-simulator  

---

## 📘 Overview

This project simulates the financial impact of major life decisions—such as going back to school, switching careers, or having a child—over a 10-year time horizon. It uses a combination of rule-based logic and machine learning regression models to estimate outcomes like net income, break-even points, and opportunity cost.

The goal: **Empower people to make informed life choices using data**.

---

## 🚀 Features

- Input key life parameters (e.g. salary, dependents, education costs)
- Predict long-term income based on different choices
- Estimate break-even years when investments start paying off
- Visualize scenarios with clean, easy-to-read charts
- Designed for usability and empathy—especially for parents, career switchers, and immigrants

---

## 🛠️ Technologies Used

- Python 3.9+
- `pandas`, `numpy`
- `scikit-learn` (regression modeling)
- `matplotlib` / `seaborn` (visualizations)
- Optionally: `streamlit` or `gradio` (for web app version)

---

## 📊 Example Use Case

> “If I go back to school for 2 years and pay $30,000 tuition, how long will it take to recover the cost and start earning more?”

Output:
- Projected net income over 10 years
- Year when the new salary surpasses the original path
- Total ROI and opportunity cost

---

## 🧠 How It Works

1. User provides life decision parameters (salary, cost, time off work, etc.)
2. A baseline income projection is generated
3. An ML model (e.g., linear regression or decision tree) estimates the alternate scenario
4. Results are compared and visualized

---

## 📁 Files

- `notebook.ipynb` – Development and visualization notebook
- `life_simulator.py` – Modular simulation logic
- `data/sample_profiles.csv` – Example test cases
- `requirements.txt` – Dependencies for setup

---

## 🧪 Getting Started

```bash
# Clone the repository
git clone https://github.com/saadatibk/life-decision-simulator.git
cd life-decision-simulator

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook notebook.ipynb
