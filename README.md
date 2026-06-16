# 🧠 Employee Attrition Risk Predictor

A production-ready HR Analytics dashboard built with **Streamlit** and **Machine Learning** to predict employee attrition risk and support proactive retention strategies.

---

## 📌 Overview

Employee turnover is one of the most significant challenges faced by organizations. This project leverages Machine Learning to identify employees who may be at risk of leaving the organization, enabling HR teams to take preventive action before attrition occurs.

The application provides an interactive dashboard where HR professionals can enter employee information and instantly receive:

* Attrition Probability (%)
* Risk Level Classification
* Prediction Outcome
* Confidence Score
* Personalized HR Recommendations

---

## 🚀 Live Features

### 👤 Employee Information

Capture key employee demographics and profile details:

* Age
* Gender
* Marital Status
* Age Group
* Distance From Home

### 💼 Job Information

* Department
* Job Role
* Education Field
* Business Travel
* Overtime Status

### 💰 Compensation

* Monthly Income
* Daily Rate
* Job Level
* Stock Option Level

### 📅 Experience Metrics

* Total Working Years
* Years At Company
* Years In Current Role
* Years Since Last Promotion
* Years With Current Manager
* Training Times Last Year
* Tenure Group

### 😊 Satisfaction Metrics

* Environment Satisfaction
* Job Satisfaction
* Relationship Satisfaction
* Job Involvement
* Work-Life Balance

---

## 🤖 Machine Learning Model

### Model Used

* Logistic Regression

### Target Variable

* Attrition

  * 0 = Employee Stays
  * 1 = Employee Leaves

### Dataset

IBM HR Analytics Employee Attrition & Performance Dataset

### Feature Engineering

The application automatically generates additional predictive features:

* YearSincePromotionRatio
* YearsWithManagerRatio
* AverageYearsperCompany
* IncomePerJoblevel

These features are generated internally and are not displayed to end users.

---

## 📊 Prediction Output

The dashboard provides:

### Prediction Result

* Likely to Stay
* Likely to Leave

### Attrition Probability

Probability score displayed as a percentage.

### Risk Categories

| Risk Level     | Probability |
| -------------- | ----------- |
| 🟢 Low Risk    | 0% - 30%    |
| 🟡 Medium Risk | 30% - 60%   |
| 🔴 High Risk   | 60% - 100%  |

### Confidence Score

Displays the model's confidence in its prediction.

---

## 💡 HR Recommendations

The system generates personalized recommendations based on predicted risk.

### High Risk

* Conduct retention discussions
* Review compensation package
* Evaluate workload and overtime
* Discuss promotion and career growth opportunities

### Medium Risk

* Monitor engagement levels
* Increase feedback sessions
* Explore development opportunities

### Low Risk

* Maintain engagement initiatives
* Continue learning and development programs
* Encourage leadership and mentoring opportunities

---

## 🎨 Dashboard Highlights

* Modern SaaS-inspired UI
* Responsive design
* Dark professional theme
* Executive-style HR dashboard
* Interactive tabs
* Dynamic risk visualization
* Personalized recommendations
* Recruiter-friendly portfolio presentation

---

## 🛠️ Technology Stack

### Frontend

* Streamlit

### Machine Learning

* Scikit-Learn
* Logistic Regression

### Data Processing

* Pandas
* NumPy

### Model Persistence

* Joblib

---

## 📂 Project Structure

```text
Employee-Attrition-Predictor/
│
├── app.py
├── attrition_model.pkl
├── requirements.txt
├── README.md
│
└── assets/
    ├── dashboard.png
    └── screenshots/
```

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/your-username/employee-attrition-predictor.git
cd employee-attrition-predictor
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📦 Requirements

```text
streamlit
pandas
numpy
scikit-learn
joblib
```

---

## 📸 Dashboard Preview

Add screenshots of:

* Dashboard Home Screen
* Input Forms
* Prediction Results
* Risk Analysis Section
* HR Recommendations

---

## 🎯 Business Value

This solution helps organizations:

* Reduce employee turnover
* Improve workforce planning
* Support data-driven HR decisions
* Identify retention risks early
* Enhance employee engagement strategies

---

## 👨‍💻 Author

Kunal Walunj

### Skills Demonstrated

* Machine Learning
* Feature Engineering
* HR Analytics
* Streamlit Development
* Data Science
* UI/UX Design
* Predictive Analytics

---

## ⭐ If you found this project useful

Consider giving the repository a star and sharing feedback.
