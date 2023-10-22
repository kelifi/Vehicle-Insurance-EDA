import pickle

import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# import packages for hyperparameters tuning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sns.set(style='whitegrid')

train = pd.read_csv('./train.csv')

num_feat = ['Age', 'Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
            'Vehicle_Damage_Yes', 'Region_Code', 'Policy_Sales_Channel']

train['Gender'] = train['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
train = pd.get_dummies(train, drop_first=True)

train = train.rename(
    columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year'] = train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years'] = train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes'] = train['Vehicle_Damage_Yes'].astype('int')

ss = StandardScaler()
train[num_feat] = ss.fit_transform(train[num_feat])

mm = MinMaxScaler()
train[['Annual_Premium']] = mm.fit_transform(train[['Annual_Premium']])
train = train.drop('id', axis=1)

for column in cat_feat:
    train[column] = train[column].astype('str')

train_target = train['Response']
train = train.drop(['Response'], axis=1)
x_train, x_test, y_train, y_test = train_test_split(train, train_target, random_state=0)

random_search = {'criterion': ['entropy', 'gini'],
                 'max_depth': [2, 3, 4, 5, 6, 7, 10],
                 'min_samples_leaf': [4, 6, 8],
                 'min_samples_split': [5, 7, 10],
                 'n_estimators': [300]}

clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator=clf, param_distributions=random_search, n_iter=2,
                           cv=4, verbose=2, random_state=101, n_jobs=-1)
model.fit(x_train, y_train)
filename = 'rf_model.sav'
pickle.dump(model, open(filename, 'wb'))
rf_load = pickle.load(open(filename, 'rb'))

y_pred = rf_load.predict(x_test)


#How to test the model below
# data = {
#     'Gender': ['Male', 'Female', 'Male', 'Female'],
#     'Age': [25, 30, 35, 40],
#     'Driving_License': [1, 1, 0, 1],
#     'Region_Code': [101, 102, 103, 104],
#     'Previously_Insured': [0, 1, 0, 1],
#     'Annual_Premium': [5000, 6000, 7000, 8000],
#     'Policy_Sales_Channel': [1, 2, 3, 4],
#     'Vintage': [200, 250, 300, 350],
#     'Vehicle_Age_lt_1_Year': [1, 0, 0, 1],
#     'Vehicle_Age_gt_2_Years': [0, 1, 0, 1],
#     'Vehicle_Damage_Yes': [1, 0, 1, 0]
# }
#
#
# filename = 'rf_model.sav'
# rf_load = pickle.load(open(filename, 'rb'))
# df = pd.DataFrame(data)
# df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
#
# y_pred = rf_load.predict(df)
# print(y_pred)