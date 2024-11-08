# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 07:38:39 2024

@author: hbehrooz
"""
import pandas as pd
import numpy as np
"""
['year', 'PIPE_ID', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'S', 'MATERIAL',
       'INSTALLED_YEAR', 'DIAM', 'TYPE', 'LENGTH', 'ZONE', 'X', 'Y', 'BREAKED',
       'BREAKS', 'TSLF', 'avg_temp', 'std_tmp', 'HDD', 'CDD', 'precipitation',
       'rain', 'snow', 'age'],
      dtype='object')
"""
df_file="breakdfv2.csv"
datatype={'year':'int', 'PIPE_ID':'int', 'A':'category', 'B':'category', 'C':'category',
          'D':'category', 'E':'category', 'F':'category', 'G':'category', 'S':'category','MATERIAL':'category',
          'INSTALLED_YEAR':'int', 'DIAM':'int', 'TYPE':'category',
          'LENGTH':'float32', 'ZONE':'category','BREAKED':'int','BREAKS':'int',
          'X':'float32', 'Y':'float32', 'TSLF':np.int32, 'avg_temp':np.float64,
          'std_tmp':np.float64, 'HDD':np.float64,
          'CDD':np.float64, 'precipitation':np.float64, 'rain':np.float64, 'snow':np.float64,'age':'int'}
data_orig=pd.read_csv(df_file,dtype=datatype)
data=data_orig[data_orig['year']<data_orig['year'].max()].copy()
#Make number of pipe a multiple of batchsize =128
#data=data[data['PIPE_ID']<100000].copy()
aa=data['PIPE_ID'].unique()
#data=data[data['PIPE_ID'].isin(aa[:((len(aa)//1280)*1280)])].copy()
data['PIPE_ID']=data['PIPE_ID'].apply(str)
data['PIPE_ID']=data['PIPE_ID'].astype('category')
#defin break filed which show at least was one break during a year (1) or not (0))
data['break']=(data['BREAKED']>0).astype(float)
data.drop(columns=['BREAKED'],inplace=True)
#normalizing the numnerical features
normalizing_features=[ 'INSTALLED_YEAR', 'DIAM', 'LENGTH','BREAKS','X', 'Y', 'TSLF', 'avg_temp',
          'std_tmp' ,'HDD','CDD', 'precipitation', 'rain', 'snow','age']
data['age']=data['year']-data['INSTALLED_YEAR']
data=data[data['age']>=0]
data=data[['MATERIAL', 'DIAM', 'TYPE', 'LENGTH', 'ZONE',
'break', 'TSLF','age','BREAKS']]
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Assuming your DataFrame is named 'data'

# Separate majority and minority classes
df_break = data[data['break'] == 1]
df_no_break = data[data['break'] == 0]

# Undersample the majority class to match the minority class size
df_no_break_under = df_no_break.sample(n=len(df_break), random_state=42)

# Combine the undersampled majority class with the minority class for training
df_train = pd.concat([df_break, df_no_break_under])

# Shuffle the training dataset to mix the records well
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# Separate the entire dataset for testing
X_test = data[['MATERIAL', 'DIAM', 'TYPE', 'LENGTH', 'ZONE', 'BREAKS', 'age', ]]#'TSLF']]
y_test = data['break']

# Separate features and target for training
X_train = df_train[['MATERIAL', 'DIAM', 'TYPE', 'LENGTH', 'ZONE', 'BREAKS', 'age',]]# 'TSLF']]
y_train = df_train['break']

# Define categorical and numerical columns
categorical_columns = ['MATERIAL', 'TYPE', 'ZONE']
numerical_columns = ['DIAM', 'LENGTH', 'BREAKS', 'age',]# 'TSLF']

# One-hot encode categorical features and standardize numerical features, with sparse output set to False
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_columns),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_columns)
])

# Set up the SVC model pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', C=1, gamma='scale', probability=True))
])

# Train the model on the balanced training dataset
pipeline.fit(X_train, y_train)

# Make predictions on the test set (pipeline applies the transformation automatically)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]


# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))
#%%
