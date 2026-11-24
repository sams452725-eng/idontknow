from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier

# 1. 데이터
x, y = load_breast_cancer(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=444, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = DecisionTreeClassifier()
# model=RandomForestClassifier()
# model=BaggingClassifier(
#     DecisionTreeClassifier(),
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
acc = accuracy_score(y_test, y_pred)
print("acc:", acc)

# model=RandomForestRegressor()
# 최종점수: 0.9298245614035088
# acc: 0.9298245614035088

# model=DecisionTreeRegressor()
# 최종점수: 0.9035087719298246
# acc: 0.9035087719298246

# model=BaggingRegressor(bootstrap=True)
# 최종점수: 0.9210526315789473
# acc: 0.9210526315789473

# model=BaggingRegressor(bootstrap=False)
# 최종점수: 0.9298245614035088
# acc: 0.9298245614035088
