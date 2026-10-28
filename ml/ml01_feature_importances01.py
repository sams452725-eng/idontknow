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
model1 = DecisionTreeClassifier(random_state=random_state)

model2 = RandomForestClassifier(random_state=random_state)

model3 = GradientBoostingClassifier(random_state=random_state)

model4 = XGBClassifier(random_state=random_state)

models = [model1,model2,model3,model4]

#3. 컴파일, 훈련
# model1.fit(x_train, y_train)

# model2.fit(x_train, y_train)

# model3.fit(x_train, y_train)

# model4.fit(x_train, y_train)

# Classifier의 score는 acc 기준이다.
for model in models:
    #3. 컴파일, 훈련
    model.fit(x_train, y_train)
    print('='*25, model.__class__.__name__, '='*25)

    #4. 평가, 예측
    score = model.score(x_test,y_test)
    print('acc : ', score)
    print(model.feature_importances_)

#4. 평가, 예측
# score = model1.score(x_test, y_test)
# print(score)       # 0.8333333333333334


# score = model2.score(x_test, y_test)
# print(score)       # 0.9333333333333333

# score = model3.score(x_test, y_test)
# print(score)       # 0.9666666666666667

# score = model4.score(x_test, y_test)
# print(score)       # 0.9333333333333333


'''
for model in models 결과
========================= DecisionTreeClassifier =========================
acc :  0.8333333333333334
========================= RandomForestClassifier =========================
acc :  0.9333333333333333
========================= GradientBoostingClassifier =========================
acc :  0.9666666666666667
========================= XGBClassifier =========================    xgboost는 통상 파라미터 튜닝이 가미되어야 결과가 잘 나온다.
acc :  0.9333333333333333


print(model.feature_importances_)
iris_datasets에 컬럼이 4개이고, feature_importances를 통해 나온 각 컬럼의 숫자가 높을수록 중요도가 높다.
숫자가 높은게 좋을수는 있지만 무조건 좋은건 아니다.
중요도가 작다고 해당 컬럼을 없애면 결과가 떨어질수있다.
왜냐하면 각 컬럼들 끼리도 상관관계가 있기 때문에 컬럼 하나가 빠졌을때 y=wx+b에서 한부분이 통으로 사라지기 때문이다.
특히 컬럼의 개수가 적을 때, 아래처럼 4개의 컬럼에서 하나를 날리는건 정말 위험하다.
해당 score = f_score, frequency_score로 acc 점수를 만들기 위해서 각 feature를 해당 %만큼 사용했다.
========================= DecisionTreeClassifier =========================
acc :  0.8333333333333334
[0.0125     0.03       0.92133357 0.03616643]
========================= RandomForestClassifier =========================
acc :  0.9333333333333333
[0.08860798 0.0214775  0.48716455 0.40274997]
========================= GradientBoostingClassifier =========================
acc :  0.9666666666666667
[0.00157033 0.02147603 0.82483537 0.15211826]
========================= XGBClassifier =========================
acc :  0.9333333333333333
[0.02430454 0.02472077 0.7376847  0.21328993]

'''