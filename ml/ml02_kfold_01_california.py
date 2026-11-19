# 맹그러봐!!!

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor,DecisionTreeRegressor,RandomForestRegressor,GradientBoostingRegressor


# 1.데이터

datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
(x_train, x_test, y_train, y_test) = train_test_split(x, y, train_size=0.7, random_state=10)
print(x_train.shape, y_train.shape)  # (14447,8) (14447,)


n_split = 5
kfold = KFold(n_splits = n_split, shuffle = True, random_state = 333)

#2. 모델
model = XGBRegressor()


#3,4 훈련(컴파일 없음), 평가
scores= cross_val_score(model, x, y, cv = kfold)  # cross_val_score가 model.fit까지 시킴

print('acc : ', scores, '\n평균 acc :', round(np.mean(scores),2))

# DecisionTreeRegressor
# acc :  [0.59432531 0.6148772  0.58737676 0.61106694 0.62434018] 
# 평균 acc : 0.61

# RandomForestRegressor
# acc :  [0.80292859 0.80580906 0.79528604 0.81891941 0.82221029] 
# 평균 acc : 0.81

# GradientBoostingRegressor
# acc :  [0.7985882  0.80773472 0.79696532 0.81836523 0.82404709] 
# 평균 acc : 0.81

# XGBRegressor
# acc :  [0.82742129 0.84140555 0.82123327 0.84284717 0.8478747 ] 
# 평균 acc : 0.84