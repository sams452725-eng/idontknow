from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

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
model = DecisionTreeRegressor()
# model=RandomForestRegressor()
# model=BaggingRegressor(
#     DecisionTreeRegressor(),
#     n_estimators=10,
#     random_state=333,
#     n_jobs=-1,
#     max_samples=0.8,
#     bootstrap=True,      # True가 디폴트
#     # bootstrap=False           # 데이터의 중복 허용X
# )

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
