import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

tf.random.set_seed(333)
np.random.seed(333)

# 1. 데이터
x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_data = np.array([0, 1, 1, 0])

# 2. 모델
# model=Perceptron()
# model=LinearSVC()
model = DecisionTreeClassifier()
# model.add(Dense(10, input_dim=2))
# model.add(Dense(1, activation='sigmoid'))


# 3. 훈련
model.fit(x_data, y_data)

# 4. 평가,예측
y_predic = model.predict(x_data)

result = model.score(x_data, y_data)
print("acc:", result)

acc = accuracy_score(y_data, np.round(y_predic))
print("accuracy_score:", acc)
