# classifier
# 맹그러봐

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.utils import all_estimators
import sklearn as sk

import warnings
warnings.filterwarnings('ignore')


#1. 데이터
x, y = load_iris(return_X_y = True)
# print(x, y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    shuffle = True,
    random_state = 333,
    stratify = y,
    train_size = 0.8,
    )

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)



#2. 모델
all = all_estimators(type_filter='classifier')
print('all_models: ', all)
print('sk version : ', sk.__version__)
print('모델의 개수: ', len(all))

# sk version :  1.9.1
# 모델의 개수:  44

# all에 2개 들어가 있으므로 앞에꺼는 name, 2번째꺼는 model로 빼옴
for name, model in all:
    # print(name)
    try:
        n_split = 5
        kfold = KFold(n_splits = n_split, shuffle = True, random_state = 333)

        model = model()   # 인스턴스 생성도 try 안에서 (estimator 필수 인자 모델은 여기서 오류)

        #3,4 훈련(컴파일 없음), 평가
        scores= cross_val_score(model, x_train, y_train, cv = kfold)  # cross_val_score가 model.fit까지 시킴

        print('acc : ', scores, '\n평균 acc :', round(np.mean(scores),2))
        # 분류: acc, 회귀: R2
        # acc :  [0.93333333 0.9        1.         0.96666667 0.93333333] 
        # 평균 acc : 0.95

        y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

        print(y_predict)
        print(y_test)

        # 맹그러바  accuracy_score

        accuracy = accuracy_score(y_test, y_predict)
        print("accuracy: ", accuracy)
    except:
        print(name, "모델 오류")