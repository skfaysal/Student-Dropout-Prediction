# importing libraries
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# importing dataset
data = pd.read_csv("./dataset/data.csv")

"""EDA"""
print(data.info)
print(data.columns)
print(data.head(10))

# Check Missing values
print(data.isnull().sum())

# drop rows which have null values
data = data.dropna()

# drop unnecessary columns
data.drop(['id', 'STUDENT_ID', 'dropoutCause', 'courses_need_to_retake', 'create_uid', 'create_date', 'write_uid',
           'write_date'], axis=1, inplace=True)

# getting the number of labels
for col in data.columns:
    print(col, ':', len(data[col].unique()), 'lebels')

# Plot correlation matrix
plt.matshow(data.corr())
plt.show()

# for better view of correlation
corr = data.corr()
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

"""ËNCODING"""
# create list of columns which needs to be encoded
cat_cols = ['facultyShortName', 'departmentShortName', 'gender_maritalStatus_calculation', 'mentors_feedback',
            'dropoutStatus']
# encode and convert it into dummy variable. here drop first meaning dropping out one dummy variable from each features.
# It helps to get rid of curse of dimensionality
encoded_data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# Saving the encoded columns for using while inference/predicting on new data.
# All columns except target variable("dropout_status") will be saved.
# Because while predicting on new data we don't pass target variable into the model
encoded_data_inference = encoded_data.iloc[:, :-1]
# Saving the columns in the disk as pkl format
filename = './saved models/encoder.pkl'
pickle.dump(encoded_data_inference.columns, open(filename, 'wb'))

# Splitting the dataset into independent(x) and dependent/target(y) variable/features
x = encoded_data.iloc[:, :-1].values
y = encoded_data.iloc[:, -1].values

# Splitting the variables/features into train and test set by 80/20 (80% for training and 20% for testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Normalizing or scalling the dataset into binomial/gaussian distribution
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# saving the scalling object for using on new data
filename = './saved models/sc.pkl'
pickle.dump(sc, open(filename, 'wb'))

"""MODEL"""
"""Logistic Regression"""
# Create object for LogisticRegression.set max_iter>100 as the dataset is big
logisticRegr = LogisticRegression(solver='lbfgs', max_iter=300)
# training
logisticRegr.fit(x_train, y_train)
# Predicting on test data
logr_predict = logisticRegr.predict(x_test)

# Accuracy on test data
print("Accuracy On Test Data:", metrics.accuracy_score(y_test, logr_predict))

# visualize confusion matrix for logistic regression
cm = metrics.confusion_matrix(y_test, logr_predict)
print(cm)
# Better view
score=metrics.accuracy_score(y_test, logr_predict)
plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15);

"""ROC and AUC for random forest"""
# calculate scores
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = logisticRegr.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='class 0')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='class 1')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()

# saving the model on the disk as pkl
filename = './saved models/model_logreg.pkl'
pickle.dump(logisticRegr, open(filename, 'wb'))

"""Random Forest"""
randomforest = RandomForestClassifier(n_estimators=20, random_state=0)
# training
randomforest.fit(x_train, y_train)
# Predicting on test data
y_pred = randomforest.predict(x_test)
print("Accuracy on test data:", metrics.accuracy_score(y_test, y_pred))

# visualize confusion matrix for Random Forest
cm = metrics.confusion_matrix(y_test, logr_predict)
print(cm)
# Better view
score=metrics.accuracy_score(y_test, y_pred)

plt.figure(figsize=(9, 9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size=15);

"""ROC and AUC for random forest"""

# calculate scores
ns_probs = [0 for _ in range(len(y_test))]
# predict probabilities
lr_probs = randomforest.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# calculate scores
ns_auc = roc_auc_score(y_test, ns_probs)
lr_auc = roc_auc_score(y_test, lr_probs)
# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Logistic: ROC AUC=%.3f' % (lr_auc))
# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='Class 0')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Class 1')
# axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# show the legend
pyplot.legend()
# show the plot
pyplot.show()
# saving the model on the disk as pkl
filename = './saved models/model_randomforest.pkl'
pickle.dump(randomforest, open(filename, 'wb'))

""""Compare two model using ROC"""

# predict probabilities
pred_prob1 = logisticRegr.predict_proba(x_test)
pred_prob2 = randomforest.predict_proba(x_test)

# roc curve for models
fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:, 1], pos_label=1)
fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:, 1], pos_label=1)

# roc curve for tpr = fpr
random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# auc scores
auc_score1 = roc_auc_score(y_test, pred_prob1[:, 1])
auc_score2 = roc_auc_score(y_test, pred_prob2[:, 1])
# matplotlib
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--', color='orange', label='Logistic Regression')
plt.plot(fpr2, tpr2, linestyle='--', color='green', label='Random Forest')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC', dpi=300)
plt.show();

# Test on new data
#load encoder
enc = 'C:/Users/Administrator/Desktop/Dropout_phase2/dropout model/encoder.pkl'
encoder = pickle.load(open(enc, 'rb'))

#load scaler
scl = 'C:/Users/Administrator/Desktop/Dropout_phase2/dropout model/sc.pkl'
scaler = pickle.load(open(scl, 'rb'))

#load model
mod = 'C:/Users/Administrator/Desktop/Dropout_phase2/dropout model/model_logreg.pkl'
model = pickle.load(open(mod, 'rb'))

mod1 = 'C:/Users/Administrator/Desktop/Dropout_phase2/dropout model/model_randomforest.pkl'
randomforest = pickle.load(open(mod, 'rb'))

test_df = pd.DataFrame({'facultyShortName': ['FSIT'],
                        'departmentShortName': ['CSE'],
                        'PreviousResult': [20],
                        'ParentsTotalIncome': [10],
                        'gender_maritalStatus_calculation': ['Negative'],
                        'no_of_drop_semester': [10],
                        'no_course_need_to_retake': [100],
                        'mentors_feedback': ['Critical'],
                        'Attendance': [10],
                        'SGPA': [20],
                        'assignment': [80],
                        'presentation': [15],
                        'dueAmount': [80],
                        })

cat_cols = ['facultyShortName', 'departmentShortName', 'gender_maritalStatus_calculation', 'mentors_feedback']
encoded_test_data = pd.get_dummies(test_df, columns=cat_cols)

encoded_test_data_m = encoded_test_data.reindex(columns=encoder, fill_value=0)
print("Encoded test data shape: ",encoded_test_data_m.shape)

a = np.array(encoded_test_data_m)
b = scaler.transform(a)
prediction = randomforest.predict_proba(b)
print(prediction)


"""
import shap

# Create object that can calculate shap values
explainer = shap.TreeExplainer(randomforest)

# Calculate Shap values
shap_values = explainer.shap_values(encoded_test_data_m)

# creating a dataframe for the features and corresponding shap values
x_shap_df = pd.DataFrame(shap_values[1], columns=encoded_data.iloc[:, :-1].columns)

# converting the row dataframe into columns dataframe
x_shap_sr = x_shap_df.max()

# order the most contributed features in descending fashoin
ordered_features = x_shap_sr.sort_values(ascending=False)

# see the most 10 contributed features with score
print(ordered_features.head(5))
"""