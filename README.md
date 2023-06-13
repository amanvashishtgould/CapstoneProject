# Capstone Project: Hospital Mortality Prediction
**Author: Aman Vashisht**

Link to Jupyter notebook: https://github.com/amanvashishtgould/CapstoneProject/blob/main/Capstone_hospital_mortality.ipynb 


#### Executive summary
This capstone project develops and cross-validates machine learning(ML) and AI models to predict whether or not a patient at a hospital ICU survives, and further aims to characterize the predictors of ICU mortality. This is done by utilizing a critical care database of ICU-admitted patients, and understanding the data, types of variables, missing values, histogram distributions of predictors, independence of predictor variables, and balancing the target survival/non-survival classes. Several models are fit and cross validated using selected evaluation metrics. Models with best performance are further optimized using hyperparameter tuning, which further slightly increased the accuracy of the chosen evaluation metrics. Lastly, the top predictors of ICU mortality are identified, and recommendations for future deployment and model improvement summarized.

#### Rationale

The predictors of in-hospital mortality for intensive care units (ICU) remain poorly characterized. Developing predictive models based on mortality risk factors can prevent mortality through controlling acute conditions and planning in intensive care units. This could help hospitals to focus their resources and care on critical patients who have these mortality risk factors, and improve ICU survival rates and overall, the healthcare system.

#### Research Question

The goal of this project is to predict whether or not a patient at a hospital ICU would survive. To this end, I aim to develop and cross-validate prediction models for all-cause in-hospital mortality among ICU-admitted patients, using different demographic, personal medical history, comorbidities, and lab (bloodwork, urine test, pulmonary tests, cardiac tests etc.) data.

#### Data Sources

The dataset used is from the Medical Information Mart for Intensive Care III (MIMIC-III) based database found on Kaggle (https://www.kaggle.com/datasets/saurabhshahane/in-hospital-mortality-prediction) . It is a publicly available critical care database containing de-identified data on patient admissions to the ICU of the Beth Israel Deaconess Medical Center, Boston, USA, between 1 June, 2001 and 31 October, 2012. In the data, the demographic characteristics and vital signs extracted were recorded during the ﬁrst 24 hours of each admission and laboratory variables were measured during the entire ICU stay. Comorbidities were identified using ICD-9 codes.

#### Methodology

- CRISP-DM framework is followed throughout this project.
- After data cleaning, a basic exploratory data analysis is performed.
- Since this is a classification machine learning (ML) problem, several ML models like Logistic Regression, Decision Trees, K Nearest Neighbors (KNN), and Support Vector machines (SVM) are fit.
- Additionally, some ensemble models like Random Forests, AdaBoost Classifier etc. are also implemented.
- Furthermore, a Neural Networks model is also explored as an additional model option for this problem.
- Feature importance is further employed to identify the variables that predict the mortality factors in the patients.

#### Results

**Business understanding**:

The goal of this project is to predict the mortality of ICU patients at a hospital (i.e., whether or not a patient at the ICU would survive).

**Data understanding**:

The dataset is a publicly available critical care database containing de-identified data on patient admissions, and includes demographic characteristics, vital signs, medical history like comorbidities, and measured laboratory variables.

**Data processing and Exploratory Data Analysis**:

There are 1177 rows and 51 columns in this dataset, with numeric(int or float) type columns.

The target variable is the outcome column, and the predictor variables are columns including age, BMI, gender, red blood cell count, white blood cell(leukocyte) count, diabetes (presence or absence), urine output, urea nitrogen level, blood chloride level, blood sodium level, blood anion gap, arterial/venous partial carbon dioxide pressure, respiration rate, heart rate, systolic and diastolic blood pressure, etc.

Some columns like BMI, heart rate, pH etc. have missing values, ranging from 1-25%. Outcome has one missing value, and its corresponding row is dropped. 
The outcome data is imbalanced as there are about 86.5 % of survivors and 13.5% non-survivors.

Histogram distributions of all the predictor variable columns are plotted and colored by the outcome variable, and it was seen that some variables like age, BMI, atrialfibrillation, platelets, blood potassium, anion gap, bicarbonate, and lactic acid do seem to show some differences in distributions for different outcomes.

**Data Preparation: Feature Engineering and Train-test split**:

A quick seaborn-based heatmap of correlation of different variables is plotted. The outcome does not seem to be highly correlated with any of the variables. There are some high correlations amongst the variables like PT and INR; MCH and MCV; hematocrit and RBC; and Lymphocytes and Neutrophils. One variable in each of these pairs of multicollinear variables are removed.

KNN Imputer was used to fill the missing values, after splitting the data into train and test sets.

Since the target data is imbalanced, Synthetic Minority Oversampling Technique (SMOTE) is utilized to balance out the target classes.

Lastly, the features are scaled/standardized using StandardScaler().

**Modeling**:

First a baseline model is run using the Dummy Classifier. Additionally, common classifier models like Logistic Regression, K Nearest Neighbor (KNN), Decision Trees, and Support Vector Machines (SVM) are fit and cross-validated using the test-set. Two ensemble models namely Random Forest and AdaBoost classifier and also utilized. Ridge Classifier is also fit and cross-validated, after perusing a package called Lazypredict’s results. Lastly, a 2 layered neural network is trained and cross-validated to see if it improved the results (-it did not). Confusion matrix is plotted, and classification reports showing different scores like Precision, Recall, F-1 are shown.

**For the critical life-saving decisions of ICU patients, it is important to minimize false negatives, so Recall is a crucial metric** in this exercise. The following summarizes the recall scores for the different models fit along with the times, test accuracies, and F-1 scores.


*Dummy Classifier:*

Test accuracy-0.86

Test F1 score-0

Test Recall score-0

Time-0.0003



*Logistic Regression:*

Test accuracy-0.79

Test F1 score-0.44

Test Recall score-0.60

Time-0.06



*KNN:*

Test accuracy-0.64

Test F1 score-0.34

Test Recall score-0.67

Time-0.003



*Decision Tree:*

Test accuracy-0.79

Test F1 score-0.29

Test Recall score-0.32

Time-0.07



*SVM:*

Test accuracy-0.83

Test F1 score-0.29

Test Recall score-0.25

Time-0.9



*Random Forest:*

Test accuracy-0.86

Test F1 score-0.34

Test Recall score-0.28

Time-0.8



*AdaBoost Classifier:*

Test accuracy-0.84

Test F1 score-0.39

Test Recall score-0.38

Time-0.6


*Ridge Classifier:*

Test accuracy-0.78

Test F1 score-0.43

Test Recall score-0.60

Time-0.001


*Neural Network with 2 layers:*

Test accuracy-0.82

Test F1 score-0.35

Test Recall score-0.37

Time-7



While a high recall is crucial to minimize false negatives of wrongly identifying patients who are critical/ will die as survivors, but it may also be important to also have a higher F-1 score and ROC AUC for a better performing model so that overall scores are high in different aspects and less time/energy/money is wasted on false positives. So it maybe important to pick a model by compromise such that all score are generally high.

Models with both a high recall (>0.6) and a reasonably high ROC AUC and F-1 scores are **Logistic Regression**  and  **Ridge Classifier**.


**Hyperparameter optimization using GridSearchCV**:

This was done for the two models that are performing best overall....i.e, Ridge regression and Logistic regression which have a recall higher than or equal to 60% and F-1 scores of about 40% in models that I ran. 

This gridsearchcv step did improve the test scores compared to the previous models. The performance of both Ridge and Logistic classification models were Recall of about 63%, ROC AUC of 0.78, and F-1 scores of 45% (improved by 1-2% in the hyperparameter optimization step).

Feature importance was derived from the coefficients of these models. From both Ridge classifier and Logistic regression, the top variables that predict the survival/mortality of a patient in ICU are summarized below.


#### Recommendations and Next steps:

Feature importance reveals similar features from both models that predict mortality at ICU in the hospital and these are (top 5 in descending order of importance):

-Chloride

-Blood Sodium

-Bicarbonate

-Anion gap

-Urea nitrogen

Next steps in the future could involve collecting additional data, imporving model fit and scores, and validating improved model on additional validation data. To improve computational efficiency, advanced packages like PyCaret could be used, and AutoML could be utilized. Deployment of the model should be done while keeping in mind the limitations of the model, and prioritizing patient care and efficient and careful utilization of ICU resources.


#### Outline of project
- Data: https://github.com/amanvashishtgould/CapstoneProject/tree/main/Data 
- Notebook: https://github.com/amanvashishtgould/CapstoneProject/blob/main/Capstone_hospital_mortality.ipynb 

##### Contact and Further Information
Aman Vashisht: amanvashishtgould@gmail.com

LinkedIn link: https://www.linkedin.com/in/aman-vashisht-phd-60252a107/ 

