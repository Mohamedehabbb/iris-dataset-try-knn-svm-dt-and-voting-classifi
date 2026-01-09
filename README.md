# iris-dataset-try-knn-svm-dt-and-voting-classifi
# 🌸 Iris Species Classification — Machine Learning Project

## 📌 Project Overview

The **Iris Species Classification** project is a classic supervised machine learning task that demonstrates an **end-to-end data science workflow**.

The goal is to build, tune, and evaluate multiple classification models that accurately predict the species of an Iris flower based on its physical measurements.

This project is designed with **production-level best practices** including:

* Clean data preprocessing pipelines
* Hyperparameter tuning
* Ensemble learning
* Model evaluation & comparison

---

## 🎯 Problem Statement & Objective

### 🔹 Problem Statement

The core problem addressed in this project is to **accurately classify Iris flower species** based on their physical measurements.

Although the Iris dataset is relatively small, it represents a **real-world multiclass classification problem** where:

* Multiple numerical features influence the target
* Classes may overlap in feature space
* Model generalization and robustness are critical

The challenge is not just achieving high accuracy, but building a **reliable, reusable, and well-structured ML pipeline** that follows best practices.

### 🔹 Objective

* Predict the correct Iris species (*Setosa, Versicolor, Virginica*)
* Compare multiple machine learning models
* Optimize performance using hyperparameter tuning
* Improve stability using ensemble learning

---

## 🧠 Dataset Description

* **Source:** Built-in Iris dataset from `scikit-learn`
* **Samples:** 150 observations
* **Features:** 4 numerical features
* **Target Classes:** 3 flower species

| Feature      | Description              |
| ------------ | ------------------------ |
| Sepal Length | Length of the sepal (cm) |
| Sepal Width  | Width of the sepal (cm)  |
| Petal Length | Length of the petal (cm) |
| Petal Width  | Width of the petal (cm)  |

---

## 🔄 Methodology & Workflow

The project follows a **structured Data Science lifecycle**, ensuring reproducibility and scalability.

### 1️⃣ Data Understanding

* Loaded the dataset using `scikit-learn`
* Inspected feature distributions and class balance
* Verified data quality (no missing or duplicate values)

### 2️⃣ Exploratory Data Analysis (EDA)

* Analyzed feature statistics and ranges
* Visualized relationships between features
* Observed clear separation for *Setosa* and overlap between *Versicolor* and *Virginica*

**Why this step matters:**
EDA helps guide model selection and highlights potential challenges such as class overlap.

### 3️⃣ Data Preprocessing

* Applied feature scaling using `StandardScaler`
* Used **Pipelines** to combine preprocessing and modeling
* Ensured no data leakage between training and testing phases

**Why this step matters:**
Scaling is essential for distance-based and margin-based models like KNN and SVM.

### 4️⃣ Model Development

The following models were trained:

* **Logistic Regression** as a simple, interpretable baseline
* **K-Nearest Neighbors (KNN)** to capture local patterns
* **Support Vector Machine (SVM)** for high-performance decision boundaries

### 5️⃣ Hyperparameter Tuning

* Implemented **GridSearchCV**
* Used cross-validation to avoid overfitting
* Selected optimal parameters for each model

### 6️⃣ Ensemble Learning

* Built a **Voting Classifier** combining top-performing models
* Achieved improved generalization and stability

### 7️⃣ Evaluation

* Accuracy score
* Confusion matrix
* Classification report

---

## ⚙️ Models & Techniques

| Model               | Purpose                                  |
| ------------------- | ---------------------------------------- |
| Logistic Regression | Baseline interpretable classifier        |
| KNN                 | Distance-based classification            |
| SVM (Tuned)         | High-performance margin-based classifier |
| Voting Classifier   | Ensemble for performance optimization    |

**Key Techniques Used:**

* Scikit-learn Pipelines
* Feature scaling
* GridSearchCV
* Cross-validation
* Ensemble learning

---

## 📈 Results & Performance

* **Logistic Regression Accuracy:** ~95%
* **Tuned KNN Accuracy:** ~96–97%
* **Tuned SVM Accuracy:** ~97–98%
* **Voting Classifier Accuracy:** **~98%**

### 🔹 Key Observations

* Ensemble learning improved consistency across classes
* Most misclassifications occurred between *Versicolor* and *Virginica*, which naturally overlap
* The final pipeline achieved strong generalization despite the small dataset

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Libraries:**

  * Pandas
  * NumPy
  * Scikit-learn
  * Matplotlib
  * Seaborn

---

## 📂 Project Structure

```
iris-ml-pipeline/
│
├── data/
│   └── iris.csv
├── notebooks/
│   └── iris_exploration.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── README.md
└── requirements.txt
```

---

## ⚠️ Challenges Faced

* Overlapping feature distributions between some classes
* Risk of overfitting due to small dataset size
* Selecting optimal hyperparameters without data leakage

### 🔹 How These Challenges Were Addressed

* Used cross-validation to ensure robustness
* Applied pipelines to prevent data leakage
* Used ensemble learning to reduce model variance

---

## 🎓 Lessons Learned

* Even simple datasets benefit from **structured ML pipelines**
* Hyperparameter tuning significantly impacts performance
* Ensemble models often outperform single classifiers
* Clean preprocessing is critical for fair model evaluation

---

## 🚀 Key Takeaways

* Demonstrates a **complete, professional ML workflow**
* Emphasizes reproducibility and best practices
* Highlights the importance of model comparison and optimization
* Serves as a strong portfolio project for **Data Scientist / ML Engineer roles**

---

## 👤 Author

**Mohamed Ehab**  
Data Scientist | Machine Learning Engineer

- 📧 Email: moehab1532002@gmail.com  
- 📱 Phone: +20 109 014 6607  
- 🔗 LinkedIn: https://www.linkedin.com/in/mohamed-ehab-7b91092b3  
- 🐙 GitHub: https://github.com/Mohamedehabbb

⭐ *This project demonstrates a professional, end-to-end approach to regression modeling with a strong focus on business impact and interpretability.*
## 🔗 Kaggle Notebook
You can view the complete notebook and full execution on Kaggle:  
👉[ https://www.kaggle.com/code/mohamedehaab/tv-marketing-sales-prediction-advanced-regression](https://www.kaggle.com/code/mohamedehaab/iris-dataset-try-knn-svm-dt-and-voting-classifi)


⭐ *If you find this project useful, feel free to star the repository and explore other projects on my GitHub profile.*
