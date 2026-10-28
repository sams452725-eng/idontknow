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


# ========================= XGBClassifier =========================
# acc :  0.9333333333333333
# F_score : [0.02430454 0.02472077 0.7376847  0.21328993]
# gain_score : {'f0': 0.09280245006084442, 'f1': 0.09439174085855484, 'f2': 2.8167145252227783, 'f3': 0.8144086599349976}
# 사용법 : xgboost로 돌리고 F_score는 빈도, gian_score는 손실함수의 개선률(점수) 측정해서 지금은 단순하게 점수가 높은게 좋다라고 인식하고 쓴다.
# 현재처럼 F_score와 gain_score의 순위가 같다면 상관없지만, 만약 순위가 다르면 gain_score의 점수가 높은쪽을 반영하는게 유리하다.