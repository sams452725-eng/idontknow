from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

# 1. 데이터
x, y = load_diabetes(return_X_y=True)

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

# Regressor는 voting(hard/soft) 파라미터 없음 → 예측값 평균
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
# 최종점수: 0.5423239579114119
# r2: 0.5423239579114119

# model=DecisionTreeRegressor()
# 최종점수: 0.11923753210241783
# r2: 0.11923753210241783

# model=BaggingRegressor(bootstrap=True)
# 최종점수: 0.4824125626990561
# r2: 0.4824125626990561

# model=BaggingRegressor(bootstrap=False)
# 최종점수: 0.5456563896484485
# r2: 0.5456563896484485

# model=VotingRegressor()
# 최종점수: 0.4957736770506931
# r2: 0.4957736770506931
