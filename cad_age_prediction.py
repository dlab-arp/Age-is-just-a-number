import pandas as pd
import numpy as np
import pathlib
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, confusion_matrix

indep_features = ["Sex", "BMI", "DM", "HTN", "Current Smoker", "EX-Smoker", "Obesity",
                   "BP", "PR", "FBS", "CR", "TG", "LDL", "HDL", "HB", "Lymph", "Neut",
                   "PLT", "Age"]

#dep_features = ["Age"]
dep_features = ["Cath"]

data_path = pathlib.Path(r"C:\Users\IYEH5X\UC_PhD\Surya_work\SAMHAR\CAD\data\Z-Alizadeh sani dataset.xlsx")



def minmax_scale(obj_df):
    obj_df = obj_df.astype(dtype=float)
    min_val = np.amin(obj_df)
    max_val = np.amax(obj_df)
    obj_df = (obj_df - min_val)/(max_val - min_val)
    return obj_df

def minmax_age(obj_df, min_age=0., max_age=100.):
    obj_df = obj_df.astype(dtype=float)
    obj_df = (obj_df - min_age)/(max_age - min_age)
    return obj_df


raw_data = pd.read_excel(data_path)
#Encode Categorical Variables
raw_data["Sex"] = np.where(raw_data["Sex"].str.contains("Male"), 1, 0)
raw_data["Obesity"] = np.where(raw_data["Obesity"].str.contains("Y"), 1, 0)

#Normalize Continuous Variables
raw_data["BMI"] = minmax_scale(raw_data["BMI"])
raw_data["BP"] = minmax_scale(raw_data["BP"])
raw_data["PR"] = minmax_scale(raw_data["PR"])
raw_data["FBS"] = minmax_scale(raw_data["FBS"])
raw_data["CR"] = minmax_scale(raw_data["CR"])
raw_data["TG"] = minmax_scale(raw_data["TG"])
raw_data["LDL"] = minmax_scale(raw_data["LDL"])
raw_data["HDL"] = minmax_scale(raw_data["HDL"])
raw_data["HB"] = minmax_scale(raw_data["HB"])
raw_data["Lymph"] = minmax_scale(raw_data["Lymph"])
raw_data["Neut"] = minmax_scale(raw_data["Neut"])
raw_data["PLT"] = minmax_scale(raw_data["PLT"])
raw_data["Age"] = minmax_age(raw_data["Age"], min_age=0., max_age=100.)

#Normalize Age
#raw_data["Age"] = minmax_age(raw_data["Age"], min_age=0., max_age=100.)
#raw_data["Age"] = minmax_age(raw_data["Age"], min_age=0., max_age=100.)
raw_data["Cath"] = np.where(raw_data["Cath"].str.contains("Cad"), 1, 0)

#Get Relevant Features
X_df  = raw_data.loc[:, indep_features]
X_nda = X_df.to_numpy()

Y_df  = raw_data.loc[:, dep_features]
Y_nda = Y_df.to_numpy()

#Test Train Split
X_train, X_test, y_train, y_test = train_test_split(X_nda, Y_nda, test_size=0.20, random_state=121)


#Train Model
#regressor = linear_model.LinearRegression()
#regressor.fit(X_train, y_train)

#Train Classifier
clf = linear_model.LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)

#Evaluate Performance
#y_pred = regressor.predict(X_test)
#r2 = r2_score(y_test, y_pred)
#mse = mean_squared_error(y_test, y_pred)
#mae = mean_absolute_error(y_test, y_pred)

debug = 1