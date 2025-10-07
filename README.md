# 🧠 Adult Income Prediction using XGBoost  

## 📘 Project Overview  
This project aims to predict whether an individual's annual income exceeds **$50,000** based on demographic and work-related attributes from the **UCI Adult Income Dataset (Census Income)**.  
The task is framed as a **binary classification** problem, and several machine learning models were implemented and compared — with **XGBoost** achieving the best overall performance.  

---

## 🎯 Objectives  
- Perform **data preprocessing** and cleaning on the Adult dataset.  
- Build and evaluate **multiple ML algorithms** to predict income category.  
- Compare model performance using standard classification metrics.  
- Identify key features contributing to higher income levels.  

---

## 🧩 Dataset Information  

**Source:** [UCI Machine Learning Repository – Adult Dataset](https://archive.ics.uci.edu/ml/datasets/adult)

**Number of instances:** ~48,842  
**Number of features:** 14  

**Features used:**  

| Feature | Description |
|----------|-------------|
| age | Age of the individual |
| workclass | Type of employer (Private, Self-emp, Govt, etc.) |
| fnlwgt | Final weight (sampling weight) |
| education | Highest level of education attained |
| education-num | Education level as a numerical value |
| marital-status | Marital status |
| occupation | Type of occupation |
| relationship | Relationship status (Husband, Wife, Own-child, etc.) |
| race | Race of the individual |
| sex | Gender |
| capital-gain | Capital gain in the last year |
| capital-loss | Capital loss in the last year |
| hours-per-week | Average working hours per week |
| native-country | Country of origin |
| **income (Target)** | >50K or ≤50K per year |

---

## ⚙️ Project Workflow  

### 1️⃣ Data Loading and Exploration  
- Imported and merged the **training** and **test** sets from the UCI repository.  
- Checked for missing values and replaced them using suitable imputation.  
- Analyzed distributions, outliers, and categorical variable frequencies using **Seaborn** and **Matplotlib**.  

### 2️⃣ Data Preprocessing  
- Encoded categorical features using **Label Encoding**.  
- Standardized numerical features using **StandardScaler**.  
- Split the dataset into training and test sets (typically 80–20).  

### 3️⃣ Model Building  
Implemented and compared the following models using **Scikit-learn** and **XGBoost**:
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- Gaussian Naive Bayes  
- Support Vector Machine (SVM)  
- **XGBoost Classifier**

### 4️⃣ Model Evaluation  
Each model was evaluated using:  
- **Accuracy Score**  
- **Precision, Recall, F1-Score**  
- **ROC-AUC Score**  
- **Confusion Matrix**  
- **Cross-validation** for generalization performance  

### 5️⃣ Results Summary  

| Model | Accuracy | F1-Score | ROC-AUC |
|--------|-----------|----------|---------|
| Logistic Regression | ~83% | 0.82 | 0.86 |
| Decision Tree | ~81% | 0.80 | 0.82 |
| Random Forest | ~85% | 0.84 | 0.88 |
| Naive Bayes | ~78% | 0.77 | 0.80 |
| SVM | ~84% | 0.83 | 0.87 |
| **XGBoost** | **~87%** | **0.86** | **0.90** |

✅ **XGBoost outperformed all other models** in both accuracy and F1-score.  

---

## 🧪 Key Insights  
- **Education, occupation, marital status, and capital gain** were among the most influential predictors of income.  
- Individuals working more than **45 hours per week** or with **higher education levels** were more likely to fall into the >$50K income bracket.  
- Proper preprocessing and feature encoding significantly improved model performance.  

---

## 💻 Tech Stack  
- **Language:** Python  
- **Libraries:**  
  - pandas, numpy  
  - scikit-learn  
  - xgboost  
  - matplotlib, seaborn  

---

## 🚀 How to Run the Project  

1. Clone the repository:  
   ```bash
   git clone https://github.com/<your-username>/adult-income-xgboost.git
   cd adult-income-xgboost
   ```
2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook
   ```bash
   jupyter notebook xgboost.ipynb
   ```
