# 🧠 Machine Learning Algorithms from Scratch

This repository contains Jupyter Notebook implementations of various **Machine Learning algorithms**, built from the ground up using Python. Some models use `TensorFlow` for deep learning components. Each notebook is backed by relevant datasets and showcases training, evaluation, and visualization.

---

## 📁 Repository Structure

| Notebook                    | Description                                                                                   | Dataset             |
| --------------------------- | --------------------------------------------------------------------------------------------- | ------------------- |
| `Lin-Regression.ipynb`      | Implements **Linear Regression** for predicting bike rental counts                            | `SeoulBikeData.csv` |
| `logistic-regression.ipynb` | Implements **Logistic Regression** for binary classification tasks                            | `magic04.data`      |
| `knn.ipynb`                 | Implements the **K-Nearest Neighbors (KNN)** algorithm                                        | `magic04.data`      |
| `naive-bayes.ipynb`         | Implements **Naive Bayes Classifier**                                                         | `magic04.data`      |
| `svm.ipynb`                 | Implements **Support Vector Machines (SVM)**                                                  | `magic04.data`      |
| `Kmeans-and-PCA.ipynb`      | Unsupervised learning using **K-Means Clustering** and **Principal Component Analysis (PCA)** | `seeds_dataset.txt` |
| `neural-network.ipynb`      | A simple **Neural Network** built using TensorFlow for classification                         | `magic04.data`      |

---

## 🔧 Requirements

* Python 3.7+
* Jupyter Notebook / JupyterLab
* TensorFlow
* NumPy
* pandas
* scikit-learn
* matplotlib / seaborn

Install dependencies via:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn
```

---

## 📊 Datasets

* **SeoulBikeData.csv**: Used for regression analysis (bike demand prediction).
* **magic04.data**: Used for classification algorithms like logistic regression, SVM, KNN, etc.
* **seeds\_dataset.txt**: Used for unsupervised learning (clustering and dimensionality reduction).

All datasets are included in the repository for convenience.

---

## 🚀 How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/your-ml-repo.git
   cd your-ml-repo
   ```

2. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

3. Open any `.ipynb` file and run the cells to explore the implementation and results.

---

## ✅ Features

* ✅ Clean, commented, and beginner-friendly code
* ✅ Covers both **supervised** and **unsupervised** learning
* ✅ Uses **real-world datasets**
* ✅ Visualization and performance evaluation included
* ✅ TensorFlow-based neural network implementation

---

## 📌 Future Work

* Add model evaluation metrics (F1 Score, ROC, etc.)
* Extend neural network for multiclass problems
* Add ensemble methods like Random Forest, XGBoost
* Implement cross-validation and hyperparameter tuning

---

## 🙌 Contribution

Pull requests are welcome! If you'd like to improve any part of the implementation or add a new model, feel free to fork and submit a PR.

---

## 👨‍💻 Author

**Dhurkesh B**
Feel free to connect on [LinkedIn](https://www.linkedin.com/) or explore more of my work!
