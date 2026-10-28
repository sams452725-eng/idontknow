# ml01-3 카피
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

#1. 데이터
x,y = load_iris(return_X_y=True)           # sklearn에서 x,y 데이터를 바로 분리하고 싶으면 return을 쓰면된다.
# print(x.shape, y.shape)                    # (150, 4) (150,)
random_state = 123

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=random_state,
    stratify=y
)

#2. 모델구성
model = XGBClassifier(random_state=random_state)

#3. 컴파일, 훈련
model.fit(x_train, y_train)
print('='*25, model.__class__.__name__, '='*25)

#4. 평가, 예측
score = model.score(x_test,y_test)
print('acc : ', score)
print('F_score :', model.feature_importances_)
gain_score = model.get_booster().get_score(importance_type='gain')      # gain_score는 xgboost에서만 나온다.
print('gain_score :', gain_score)
# Classifier의 score는 acc 기준이다.

import matplotlib.pyplot as plt
from xgboost import plot_importance

#5. 그래프 그리기
# F_score Graph
# plot_importance(model, importance_type='weight')
# plt.title('F_score')
# plt.show()

# Gain_score Graph
plot_importance(model, importance_type='gain')
plt.title('Gain_score')
plt.show()