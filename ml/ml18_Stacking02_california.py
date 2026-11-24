from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
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
    # stratify=y        # 회귀(y가 연속값)에서는 stratify 사용 불가
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
xgb = XGBRegressor()
lg = LGBMRegressor(verbosity=-1)
rf = RandomForestRegressor()
cat = CatBoostRegressor(verbose=0)

model = StackingRegressor(
    estimators=[("XGB", xgb), ("lg", lg), ("CAT", cat), ("RF", rf)],
    final_estimator=CatBoostRegressor(verbose=0),
    n_jobs=-1,
    cv=5,
)


# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가,예측
y_pred = model.predict(x_test)
score_new = r2_score(y_test, y_pred)
print("스태킹 결과:", score_new)


# for문으로 했을 때
# 스태킹 결과: 0.7836641233423693

# model=StackingRegressor() 사용
# 스태킹 결과: 0.8509241693435622
