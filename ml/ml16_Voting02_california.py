from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# 1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=444,
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor()
rf = RandomForestRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(
    estimators=[("XGB", xgb), ("RF", rf), ("CAT", cat), ("LG", lg)],
)


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
result = model.score(x_test, y_test)
print("최종점수:", result)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
print("r2:", r2)

# model=RandomForestRegressor()
# 최종점수: 0.6087773659242851
# r2: 0.6087773659242851

# model=DecisionTreeRegressor()
# 최종점수: 0.423442213481637
# r2: 0.423442213481637

# model=BaggingRegressor(bootstrap=True)
# 최종점수: 0.5810164233189883
# r2: 0.5810164233189883

# model=BaggingRegressor(bootstrap=False)
# 최종점수: 0.5781195164864741
# r2: 0.5781195164864741

# model=VotingRegressor()
# 최종점수: 0.8418909388639988
# r2: 0.8418909388639988
