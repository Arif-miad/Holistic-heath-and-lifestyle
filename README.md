

---

# 🌿 Holistic Health & Lifestyle Dashboard

## **Project Overview**

The **Holistic Health & Lifestyle Dashboard** is an interactive Python application built with **Streamlit** that allows users to analyze and predict overall health based on lifestyle and behavioral factors. The project uses a **synthetic yet realistic dataset** to simulate individuals’ health metrics and evaluate:

* **Overall Health Score (0–100)**
* **Health Status (Poor, Average, Good)**

This app integrates **Exploratory Data Analysis (EDA), Regression, Classification**, and **interactive visualization**, providing a complete end-to-end experience for both data exploration and predictive insights.

---

## **Key Features**

1. **Dataset Upload:** Upload your CSV dataset of holistic health metrics.
2. **EDA:**

   * Distribution plots for numeric features
   * Correlation heatmaps
   * Health status counts
3. **Regression:** Predict overall health score using Linear Regression & Random Forest Regressor
4. **Classification:** Predict health status (Poor / Average / Good) using Random Forest Classifier
5. **Interactive Dashboard:** Input lifestyle metrics and see predicted health score & status instantly, along with radar chart visualization.
6. **Feature Importance:** Identify key lifestyle factors affecting health outcomes.

---

## **Dataset Description**

The dataset simulates holistic health metrics for individuals. Each row represents a person, with the following columns:

| Column Name          | Description                                         |
| -------------------- | --------------------------------------------------- |
| Physical_Activity    | Minutes of daily moderate/vigorous activity (0–120) |
| Nutrition_Score      | Diet quality score (0–10)                           |
| Stress_Level         | Self-reported stress level (1–10)                   |
| Mindfulness          | Minutes/day of mindfulness/meditation (0–60)        |
| Sleep_Hours          | Average sleep hours/night (3–10)                    |
| Hydration            | Daily water intake in liters (0.5–5)                |
| BMI                  | Body Mass Index (18–40)                             |
| Alcohol              | Units of alcohol consumed per week (0–20)           |
| Smoking              | Cigarettes per day (0–30)                           |
| Overall_Health_Score | Composite health score (0–100)                      |
| Health_Status        | Health category label: Poor / Average / Good        |

---

## **Tech Stack**

* Python 3.x
* Streamlit
* Pandas, NumPy
* Matplotlib, Seaborn, Plotly
* scikit-learn (Regression & Classification models)

---

## **Installation & Running**

1. Clone the repository:

```bash
git clone https://github.com/Arif-miad/Holistic-heath-and-lifestyle.git
cd holistic_health_project
```

2. Install required packages:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn plotly
```

3. Run the Streamlit app:

```bash
streamlit run holistic_health_dashboard.py
```

4. Upload your CSV dataset and explore the tabs: **EDA, Regression, Classification, Interactive Dashboard**.

---

## **Screenshots**

### **Home Page**

![Home](https://github.com/Arif-miad/Holistic-heath-and-lifestyle/blob/main/holistic_health_project/images/home.PNG)

### **Regression Tab**

![Regression](https://github.com/Arif-miad/Holistic-heath-and-lifestyle/blob/main/holistic_health_project/images/regre.PNG)

### **Classification Tab**

![Classification](https://github.com/Arif-miad/Holistic-heath-and-lifestyle/blob/main/holistic_health_project/images/class.PNG)

---

## **Usage**

* **EDA tab:** Explore distributions, correlations, and health status counts.
* **Regression tab:** Predict overall health score using Linear Regression & Random Forest models.
* **Classification tab:** Predict health status and visualize feature importance.
* **Interactive Dashboard:** Input lifestyle metrics with sliders to see predicted health score & status, along with radar chart visualization.

---

## **Future Improvements**

* Color-coded health score (Red / Orange / Green) based on thresholds.
* Personalized recommendations based on user input.
* Integration with **real healthcare datasets**.
* Export results to CSV or PDF for users.

---

## **Author**

**Arif**

* kaggle : [Arif-miah](https://www.kaggle.com/miadul)

---

## **License**

This project is licensed under the MIT License.

---

