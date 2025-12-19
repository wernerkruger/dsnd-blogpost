# Predicting Student Performance on Mathematical Problems

## Motivation

This project aims to predict whether a student will correctly solve a mathematical problem step on their first attempt. This prediction is crucial for intelligent tutoring systems to:

- **Personalize learning experiences**: Adjust difficulty and provide appropriate support
- **Optimize learning paths**: Identify when students need additional practice or can move forward
- **Improve educational outcomes**: Help students learn more efficiently by providing timely interventions
- **Reduce time to mastery**: Save millions of student hours by optimizing the learning process

The project follows the CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology to systematically explore, clean, model, and evaluate student performance data.

## Libraries Used

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and evaluation metrics
  - `RandomForestClassifier`
  - `GradientBoostingClassifier`
  - `LogisticRegression`
  - `train_test_split`, `LabelEncoder`
  - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`
  - `classification_report`, `confusion_matrix`

## Files in the Repository

### Data Files
- `data/algebra_2008_2009/algebra_2008_2009_train.txt`: Training dataset (first 500 rows used for this analysis)
- `data/algebra_2008_2009/algebra_2008_2009_test.txt`: Test dataset
- `data/algebra_2008_2009/algebra_2008_2009_submission.txt`: Submission format file

### Code Files
- `analyze.ipynb`: Jupyter notebook containing the complete analysis following CRISP-DM process:
  - **Exploratory Data Analysis**: Distribution plots, histograms, correlation analysis, categorical feature analysis
  - **Data Cleaning**: Handling missing values, feature engineering
  - **Model Training**: Comparison of Logistic Regression, Random Forest, and Gradient Boosting models
  - **Model Evaluation**: Accuracy, precision, recall, F1 score, ROC AUC metrics
  - **Prediction Scenarios**: Real-world scenarios demonstrating model predictions

## Summary of Results

### Key Findings

1. **Data Characteristics**:
   - Dataset contains 500 student-step interactions with 23 features
   - Target variable (Correct First Attempt) shows class distribution that varies by student performance
   - Several features have missing values that were handled through imputation
   - Many numerical features show skewed distributions

2. **Feature Engineering**:
   - Created derived features: Total Attempts, Hint Ratio, Error Ratio
   - Extracted knowledge component counts and indicators
   - Encoded categorical variables (Student ID, Problem Name, Step Name)

3. **Model Performance**:
   - Tested three machine learning algorithms: Logistic Regression, Random Forest, and Gradient Boosting
   - Best model selected based on ROC AUC score
   - Model demonstrates ability to predict student success probability
   - Feature importance analysis reveals which factors most influence student performance

4. **Model Evaluation Metrics**:
   - **Accuracy**: Overall correctness of predictions
   - **Precision**: Of predicted successes, how many were actually correct
   - **Recall**: Of actual successes, how many were correctly identified
   - **F1 Score**: Balanced measure of precision and recall
   - **ROC AUC**: Model's ability to distinguish between classes

### Model Interpretation

The trained model can help educators and intelligent tutoring systems:

- **Identify struggling students early**: Predict low probability of success and provide proactive support
- **Personalize learning paths**: Adjust problem difficulty based on predicted performance
- **Optimize resource allocation**: Focus hints and support where most needed
- **Track learning progress**: Monitor how student performance changes over time

### Prediction Scenarios

The notebook includes two prediction scenarios:

1. **Individual Student Scenario**: Predicts success probability for a specific student attempting a problem step with given characteristics
2. **Student Profile Comparison**: Compares predictions across different student profiles (struggling, average, strong students)

These scenarios demonstrate how the model can be used in real-world educational settings to make data-driven decisions about student support and learning path optimization.

## How to Run

1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

2. Open the Jupyter notebook:
   ```bash
   jupyter notebook analyze.ipynb
   ```

3. Run all cells sequentially to reproduce the analysis

## Data Citation

If you use this dataset in your research, please cite:

Stamper, J., Niculescu-Mizil, A., Ritter, S., Gordon, G.J., & Koedinger, K.R. (2010). Algebra 2008-2009. Challenge data set from KDD Cup 2010 Educational Data Mining Challenge. Find it at http://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp

## Acknowledgments

- **Data Source**: KDD Cup 2010 Educational Data Mining Challenge
- **Project Framework**: Udacity Data Scientist Nanodegree Program
- **Methodology**: CRISP-DM (Cross-Industry Standard Process for Data Mining)

## Future Improvements

Potential areas for enhancement:

1. **Larger Dataset**: Use full dataset instead of first 500 rows for more robust model
2. **Feature Engineering**: Explore temporal features, student history, and problem difficulty metrics
3. **Advanced Models**: Experiment with neural networks, XGBoost, or ensemble methods
4. **Hyperparameter Tuning**: Optimize model parameters using grid search or Bayesian optimization
5. **Cross-Validation**: Implement k-fold cross-validation for more reliable performance estimates
6. **Feature Selection**: Use techniques like recursive feature elimination to identify optimal feature subsets
7. **Class Imbalance**: Address class imbalance if present using SMOTE or other techniques

