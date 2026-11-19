import numpy as np
from sklearn.datasets import fetch_california_housing,load_diabetes
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,mean_squared_error
from sklearn.utils import all_estimators
import sklearn as sk
import warnings
warnings.filterwarnings('ignore')
import time

#1. 데이터
california = fetch_california_housing(return_X_y = True)
# print(iris)
diabetes = load_diabetes(return_X_y = True)


datasets = [california, diabetes]
data_name = ['캘리포니아','디아벳']

#2. 모델
all = all_estimators(type_filter='regressor')

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

start = time.time()
for index, value in enumerate(datasets):
    x,y=value
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        shuffle = True,
        random_state = 333,
        # stratify = y,
        train_size = 0.8,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)


    print('='*100, data_name[index],'='*100)
    model_name=[]
    model_acc=[]


    for name, model in all:
        # print(model)
        try:
            model = model()

            #3. 훈련
            # model.fit(x_train, y_train)
            scores = cross_val_score(model, x_train, y_train, cv=kfold)

            #4. 평가
            y_predict = cross_val_predict(model, x_test, y_test, cv = kfold)

            acc = np.round(accuracy_score(y_test, y_predict),4)
            model_name.append(name)
            model_acc.append(acc)
            print(name, '의 정답률 :', acc)

        except:
            # print(name, "모델 오류")
            continue


    model_acc=np.array(model_acc)
    max_index=np.argmax(model_acc)
    print('최고 모델:', model_name[max_index], 'ACC:', model_acc[max_index])
end = time.time()

print('걸린시간:', round(end - start,2), '초')












































def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))