from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
#Load the dataframe
df_hp = pd.read_csv("train.csv")
df_hp.head()

#Convert object dtype to category

#load data into DMatrix
X, y = df_hp.iloc[:,1:-1], df_hp.iloc[:,:-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 123)
DM_train = xgb.DMatrix(data = X_train, label = y_train, enable_categorical = True)
DM_test = xgb.DMatrix(data = X_test, label = y_test, enable_categorical = True)