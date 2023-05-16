# Capstone Project: Hospital Mortality Prediction
**Author: Aman Vashisht**

Link to Jupyter notebook: https://github.com/amanvashishtgould/CapstoneProject/blob/main/Capstone_hospital_mortality.ipynb 

#### Executive summary
TBD

#### Rationale
Why should anyone care about this question?

The predictors of in-hospital mortality for intensive care units (ICU) remain poorly characterized. Developing predictive models based on mortality risk factors can positively prevent mortality through controlling acute conditions and planning in intensive care units. This could help hospitals to focus their resources and care on critical patients who have these mortality risk factors, and improve ICU survival rates and overall, the healthcare system.

#### Research Question
What are you trying to answer?

The goal of this project is to predict whether or not a patient at a hospital ICU would survive. To this end, I aim to develop and validate prediction models for all-cause in-hospital mortality among ICU-admitted patients, using different demographic, personal medical history, comorbidities, and lab (bloodwork, urine test, pulmonary tests, cardiac tests etc.) data.

#### Data Sources
What data will you use to answer you question?

The dataset used is obtained from the MIMIC-III database found on Kaggle (https://www.kaggle.com/datasets/saurabhshahane/in-hospital-mortality-prediction) . It is a publicly available critical care database containing de-identified data on patient admissions to the ICU of the Beth Israel Deaconess Medical Center, Boston, USA, between 1 June, 2001 and 31 October, 2012. In the data, the demographic characteristics and vital signs extracted were recorded during the ﬁrst 24 hours of each admission and laboratory variables were measured during the entire ICU stay. Comorbidities were identified using ICD-9 codes.

#### Methodology
What methods are you using to answer the question?

- CRISP-DM framework will be followed throughout this project.
- After data cleaning, a basic exploratory data analysis will be performed.
- Given the many variables, Principal Component Analysis (PCA) may be explored for the purposes of dimensionality reduction.
- Since this is a classification machine learning (ML) problem, several ML models like Logistic Regression, Decision Trees, K Nearest Neighbors (KNN), and Support Vector machines (SVM) will be fit.
- Additionally, some ensemble models like Random Forests, XGB Classifier, AdaBoost Classifier etc. will also be tried after they are covered in the program.
-Furthermore, Neural Networks models will also be explored as a ML model option for this problem.
- Permutation importance will be further employed to identify the variables that predict what drives the mortality in the patients.
- SHAP (SHapley Additive exPlanations) analysis may further be carried out to explain causal effects (such as how each model feature has contributed to an individual prediction), as time allows.

#### Results
What did your research find?

*Business understanding*:

The goal of this project is to predict the mortality of ICU patients at a hospital (i.e., whether or not a patient at the ICU would survive).

*Data understanding*:

The dataset is a publicly available critical care database containing de-identified data on patient admissions, and includes demographic characteristics, vital signs, medical history like comorbidities, and measured laboratory variables.

*Dara processing and Exploratory Data Analysis*:

There are 1177 rows and 51 columns in this dataset, with numeric(int or float) type columns.

The target variable is the outcome column, and the predictor variables are columns including age, BMI, gender, red blood cell count, white blood cell(leukocyte) count, diabetes (presence or absence), urine output, urea nitrogen level, blood chloride level, blood sodium level, blood anion gap, arterial/venous partial carbon dioxide pressure, respiration rate, heart rate, systolic and diastolic blood pressure, etc.

Some columns like BMI, heart rate, pH etc. have missing values, ranging from 1-25%. Outcome has one missing value, and its corresponding row is dropped. 
The outcome data is imbalanced as there are about 86.5 % of survivors and 13.5% non-survivors.

Histogram distributions of all the predictor variable columns are plotted and colored by the outcome variable, and it was seen that some variables like age, BMI, atrialfibrillation, platelets, blood potassium, anion gap, bicarbonate, and lactic acid do seem to show some differences in distributions for different outcomes.

*Data Preparation: Feature Engineering and Train-test split*:

A quick seaborn-based heatmap of correlation of different variables is plotted. The outcome does not seem to be highly correlated with any of the variables There are some high correlations amongst the variables like PT and INR; MCH and MCV; hematocrit and RBC; and Lymphocytes and Neutrophils. One variable in each of these pairs of multicollinear variables are removed. 
KNN Imputer was used to fill the missing values, after splitting the data into train and test sets. 
Since the target data is imbalanced, Synthetic Minority Oversampling Technique (SMOTE) is utilized to balance out the target classes. 
Lastly, the features are scaled/standardized using StandardScaler().

*Modeling*:

First a baseline model is run using the Dummy Classifier. Additionally, common classifier models like Logistic Regression, K Nearest Neighbor (KNN), Decision Trees, and Support Vector Machines (SVM) are fit and cross-validated using the test-set. Two additional models namely Ridge Classifier and Random Forests are also fit and cross-validated, after perusing a package called Lazypredict’s results (discussed below). Confusion matrix is plotted, and classification reports showing different scores like Precision, Recall, F-1 are shown.

For the critical life-saving decisions of ICU patients, it is important to minimize false negatives, so Recall is a crucial metric in this exercise. The following summarizes the recall scores for the different models fit along with the times, test accuracies, and F-1 score.


Dummy Classifier:

Test accuracy-0.85

Test F1 score-0

Test Recall score-0

Time-0.0003



Logistic Regression:

Test accuracy-0.64

Test F1 score-0.38

Test Recall score-0.82

Time-0.02



KNN:

Test accuracy-0.50

Test F1 score-0.31

Test Recall score-0.82

Time-0.01



Decision Tree:

Test accuracy-0.36

Test F1 score-0.28

Test Recall score-0.90

Time-0.08



SVM:

Test accuracy-0.77

Test F1 score-0.37

Test Recall score-0.50

Time-0.11



Random Forest:

Test accuracy-0.33

Test F1 score-0.28

Test Recall score-0.95

Time-0.55



Ridge Classifier:

Test accuracy-0.66

Test F1 score-0.40

Test Recall score-0.82

Time-0.01



Lazy Predict package is additionally used as it can run several ML models in a few lines of code. So instead of running models manually, lazypredict classifier is fit and scored during cross-validation for scores like ROC AUC, accuracy, F1, and Recall. Among several models run using lazypredict classifier, it is hard to determine the best model with different important metrics like Recall, ROC AUC, and F-1 score. **Random Forest classifier has the best recall (>0.95), but its F-1 score is very low**.

While a high recall is crucial to minimize false negatives of wrongly identifying patients who are critical/ will die as survivors, but it may also be important to also have a higher F-1 score and ROC AUC for a better performing model so that overall scores are high in different aspects and less time/energy/money is wasted on false positives. So it maybe important to pick a model by compromise such that all score are generall high (e.g., higher than 70%). Model with high ROC AUC and F-1 score and a reasonably high recall is **Ridge Classifier**. Of the models I ran, **Logistic regression** also fits this criterion.


*Hyperparameter optimization using GridSearchCV*:

This was done for the two models that are performing best overall....i.e, Ridge regression and Logistic regression which have a recall higher than or equal to 82% and F-1 scores of about 40% in models that I ran. This did not improve the test scores compared to models above. The ROC AUC scores of both these models are 0.79.

Feature importance was derived from the coefficients of these models. From Ridge classifier, the top 5 variables that predict the survival/mortality of a patient in ICU are:  Chloride, Blood Sodium, Bicarbonate, Anion gap, and Urea nitrogen. From Logistic Regression, the top 5 variables that predict the survival/mortality of a patient in ICU are similar to Ridge Classifier:  Chloride, Blood Sodium, Bicarbonate, Anion gap, and Urea nitrogen.


#### Next steps
What suggestions do you have for next steps?

Next steps after Module 20 would be to improve the modeling efforts to get higher F-1 and ROC AUC scores. This will be discussed with the learning facilitator, Savio. Additionally, discrepancy between F-1 scores for a given model obtained with lazypredict classifier and the model I ran will be a question (for the Consultation call).

Additionally, neural network based models will be explored. 

And lastly, SHAP (SHapley Additive exPlanations) analysis may further be carried out to explain causal effects (such as how each model feature has contributed to an individual prediction), as time allows.


#### Outline of project
- Data: https://github.com/amanvashishtgould/CapstoneProject/tree/main/Data 
- Notebook: https://github.com/amanvashishtgould/CapstoneProject/blob/main/Capstone_hospital_mortality.ipynb 

##### Contact and Further Information
Aman Vashisht: amanvashishtgould@gmail.com
LinkedIn link: https://www.linkedin.com/in/aman-vashisht-phd-60252a107/ 

