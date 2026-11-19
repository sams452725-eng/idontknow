import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data=pd.DataFrame([[2,np.nan,6,8,10, 8],
                   [2, 4, np.nan, 8, np.nan, 4],
                   [2, 4, 6, 8, 10, 12],
                   [np.nan, 4, np.nan, 8, np.nan, 8]
                   ]).T
data.columns=['x1', 'x2', 'x3', 'x4']
# print(data)

imputer=SimpleImputer(strategy='most_frequent')
data2=imputer.fit_transform(data)
print(data2)

imputer2=SimpleImputer(strategy='constant', fill_value=777)
data3=imputer2.fit_transform(data)
print(data3)

# def 함수():
#     return 1
# imputer3=SimpleImputer(strategy=함수)       # 파라미터 안에 함수 입력 가능 but 매번 정의를 해야되서 번거롭다. 그래서 lambda 사용

imputer3=SimpleImputer(
    # strategy=lambda x: np.max(x)
    strategy=lambda x: np.min(x)
)
data4=imputer3.fit_transform(data)
print(data4)
