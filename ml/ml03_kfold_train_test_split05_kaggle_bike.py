from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_score,cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler



#1. 데이터
path = './_data/kaggle_bike/'

train_csv = pd.read_csv(path + 'train.csv', index_col=0)
# print(train_csv)          #[10886 rows x 11 columns]

test_csv = pd.read_csv(path + 'test.csv', index_col=0)
# print(test_csv)           #[6493 rows x 8 columns]

submission = pd.read_csv(path + 'sampleSubmission.csv', index_col=0)
# print(submission)         #[6493 rows x 1 columns]

# print(train_csv.shape)              #[6493 rows x 1 columns]
# print(test_csv.shape)               #(6493, 8)
# print(submission.shape)             #(6493, 1)



# print(train_csv.info())          # 결측치 없음
# print(test_csv.info())           # 결측치 없음

# print(train_csv.describe())
############ 결측치 확인 ############
train_csv.isna().sum()
# print(train_csv.isna().sum())
# print(test_csv.isnull().sum())






############ x,y 분리 ############
x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
# print(x, x.shape)            #(10886, 8)

y = train_csv['count']
# print(y, y.shape)            # (10886,)


(x_train, x_test, y_train, y_test) = train_test_split(x, y, train_size=0.7, random_state=10)
# print(x_train.shape, y_train.shape)  #(7620, 8) (7620,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits = n_split, shuffle = True, random_state = 333)

#2. 모델
model = XGBRegressor()


#3,4 훈련(컴파일 없음), 평가
scores= cross_val_score(model, x_train, y_train, cv = kfold)  # cross_val_score가 model.fit까지 시킴

print('acc : ', scores, '\n평균 acc :', round(np.mean(scores),2))

# XGBRegressor
# acc :  [0.33359152 0.31047136 0.33931798 0.33344597 0.33251733] 
# 평균 acc : 0.33


y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)
r2 = r2_score(y_test, y_predict)
print('r2_score :', r2)