# Student Performance & Insurance Charges Prediction

A collection of two machine learning projects demonstrating exploratory data analysis, feature preprocessing, and regression modeling in Python.

- **Student Performance Prediction:** Estimate a student's Performance Index using multiple linear regression.
- **Insurance Charges Prediction:** Predict insurance charges and compare linear regression with degree-2 and degree-3 polynomial regression.

## Highlights

- Data cleaning, categorical encoding, and visual exploration.
- Reproducible train/test splits using `random_state=42`.
- Evaluation with MAE, MSE, RMSE, and R².
- Five-fold cross-validation for the student performance model.
- Polynomial degree comparison for the insurance charges model.

## Contents

- [Student Performance Prediction](#1-student-performance-prediction)
- [Insurance Charges Prediction](#2-insurance-charges-prediction)
- [Project Structure](#project-structure)
- [Technologies](#technologies)
- [Getting Started](#getting-started)
- [Evaluation Notes](#evaluation-notes)
- [Future Improvements](#future-improvements)

## Projects at a Glance

| Project | Notebook | Dataset | Target | Model |
| --- | --- | --- | --- | --- |
| Student Performance Prediction | `LinearRegression.ipynb` | `Student_Performance.csv` | Performance Index | Multiple linear regression |
| Insurance Charges Prediction | `PolynomialReg.ipynb` | `insurance.csv` | Charges | Polynomial regression, with degrees 1–3 compared |

## 1. Student Performance Prediction

### Objective

Predict a student's Performance Index using study habits, previous scores, and extracurricular participation.

### Dataset

The dataset contains **10,000 rows and 6 columns**. Removing 127 duplicate records leaves **9,873 rows**. There are no missing values in the supplied dataset.

| Column | Role |
| --- | --- |
| Hours Studied | Input feature |
| Previous Scores | Input feature |
| Extracurricular Activities | Input feature; encoded as No = 0 and Yes = 1 |
| Sleep Hours | Input feature |
| Sample Question Papers Practiced | Input feature |
| Performance Index | Prediction target |

### Workflow

1. Inspect data types, duplicates, missing values, and unique values.
2. Remove duplicate records.
3. Explore feature distributions and relationships using count plots, histograms, a pie chart, bar plots, a line plot, and a scatter plot.
4. Encode extracurricular participation and examine a correlation heatmap.
5. Split the data into 80% training and 20% testing sets with `random_state=42`.
6. Fit a `LinearRegression` model and evaluate its predictions.
7. Compare training and testing scores, inspect actual versus predicted values, and perform five-fold cross-validation.
8. Predict the Performance Index for an example student.

### Recorded Results

The following values are taken from the notebook's saved outputs.

| Metric | Value |
| --- | ---: |
| Test MAE | 1.647 |
| Test MSE | 4.306 |
| Test RMSE | 2.075 |
| Test R² | 0.9884 |
| Training R² | 0.9887 |
| Mean five-fold cross-validation R² | Approximately 0.9887 |

The model explains approximately **98.8% of the variation** in the test target. Previous scores have the strongest positive correlation with Performance Index in the notebook's analysis. Training and testing R² are close, and cross-validation scores are consistent across folds.

For the example input below, the saved prediction is approximately **92.11**:

```python
# Feature order: hours studied, previous scores, extracurricular
# participation, sleep hours, sample question papers practiced
student = np.array([[9, 95, 0, 7, 2]])
model.predict(student)
```

## 2. Insurance Charges Prediction

### Objective

Predict insurance charges from demographic and lifestyle features, and investigate whether polynomial terms improve predictions over a linear model.

### Dataset

The dataset contains **1,338 rows and 7 columns**. Removing one duplicate record leaves **1,337 rows**. There are no missing values in the supplied dataset.

| Column | Role |
| --- | --- |
| age | Input feature |
| sex | Input feature |
| bmi | Input feature |
| children | Input feature |
| smoker | Input feature |
| region | Input feature |
| charges | Prediction target |

### Workflow

1. Inspect the dataset and remove the duplicate record.
2. Create BMI categories and age groups for exploration.
3. Analyze distributions and compare charges across smoking status, age, BMI, region, and sex.
4. Encode sex as male = 1 and female = 0, and smoking status as yes = 1 and no = 0.
5. Apply dummy encoding to region and BMI category using `drop_first=True`.
6. Remove the derived age-group column while retaining numerical age.
7. Split the data into 80% training and 20% testing sets with `random_state=42`.
8. Standardize features using `StandardScaler`, fitted only on the training set.
9. Generate degree-2 polynomial features and fit `LinearRegression` on the transformed features.
10. Evaluate predictions and compare polynomial degrees 1, 2, and 3.

### Recorded Results

The comparison below is taken from the notebook's saved outputs. Feature counts include the bias column generated by `PolynomialFeatures`.

| Polynomial degree | Expanded features | Training R² | Test R² | Test MAE | Test RMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 — Linear | 12 | 0.738 | 0.803 | 4,334.481 | 6,023.856 |
| 2 | 78 | 0.862 | **0.902** | **2,394.833** | **4,237.946** |
| 3 | 364 | 0.878 | 0.874 | 2,963.227 | 4,803.374 |

The degree-2 model has a saved test MSE of **17,960,182.825**. It performs best among the three degrees on this test split, reducing MAE by approximately **44.7%** compared with degree 1. Degree 3 improves the training score but worsens the test score, suggesting overfitting.

The exploratory analysis finds a strong association between smoking and higher charges, with BMI patterns differing by smoking status. These are associations in the dataset, rather than evidence of causation.

## Project Structure
Student-Performance-and-Insurance-Charges-Prediction-ML-Models/
│
├── LinearRegression.ipynb
├── PolynomialReg.ipynb
├── README.md
├── Student_Performance.csv
└── insurance.csv

## Technologies

- Python
- pandas and NumPy for data preparation and numerical operations
- Matplotlib and seaborn for visualization
- scikit-learn for preprocessing, regression, and evaluation
- Jupyter Notebook or Google Colab for running the notebooks

## Getting Started

### Run locally with Jupyter

1. Clone this repository using its GitHub URL, then open a terminal in the cloned directory. Alternatively, select **Code → Download ZIP** on GitHub and extract the files.
2. Install the required packages:

   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn notebook
   ```

3. The supplied notebooks use Google Colab paths. For local execution, change their dataset-loading cells to:

   ```python
   # LinearRegression.ipynb
   df = pd.read_csv('Student_Performance.csv')

   # PolynomialReg.ipynb
   df = pd.read_csv('insurance.csv')
   ```

4. Start Jupyter from that directory:

   ```bash
   jupyter notebook
   ```

5. Open either notebook and run all cells in order.

### Run with Google Colab

1. Upload a notebook to Google Colab.
2. Upload its corresponding CSV using the Colab Files panel.
3. Keep the existing `/content/Student_Performance.csv` or `/content/insurance.csv` path.
4. Run all cells from top to bottom. Install any missing packages using `%pip install` in a notebook cell.

## Evaluation Notes

MAE and RMSE measure error in the target's units; smaller values are better. MSE measures squared error, while R² measures performance relative to predicting the target mean; values closer to 1 indicate a better fit.

The reported results are taken from the outputs saved in the notebooks. Package versions are not pinned, so results may vary across environments, especially for polynomial regression with redundant feature columns. Error magnitudes should be compared within each project because the targets have different scales.

The insurance project compares model degrees on one test split. A stronger future evaluation would select the degree through cross-validation on training data and reserve a separate test set for final assessment.

## Future Improvements

- Package preprocessing and regression in scikit-learn pipelines.
- Add residual plots and investigate large prediction errors.
- Use cross-validation and regularization such as Ridge for polynomial model selection.
- Record dependency versions and save fitted models for reuse.
- Record dependency versions for reproducible execution.

## Dataset Attribution

| Project | Local dataset | Kaggle reference |
| --- | --- | --- |
| Student Performance Prediction | `Student_Performance.csv` | [Student Performance — Multiple Linear Regression](https://www.kaggle.com/code/manarmohamed24/student-performance-multiple-linear-regression) |
| Insurance Charges Prediction | `insurance.csv` | [Medical Cost Personal Dataset](https://www.kaggle.com/datasets/d3lhomi10/medical-cost-personal-dataset) |

The student performance link points to a Kaggle notebook supplied as the project reference. Its **Input** section can be used to locate the underlying dataset. The insurance link points directly to a Kaggle dataset page.

Credit belongs to the respective dataset creators and Kaggle contributors. Consult the original dataset pages for their license terms; those terms govern reuse and redistribution of the CSV files.
