import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-5,5,0.1)

# def mish(x):
#     return x*np.tanh(np.log(1+np.exp(x)))

mish=lambda x: x*np.tanh(np.log(1+np.exp(x)))

y=mish(x)

plt.plot(x,y)
plt.grid()
plt.show()


# mish는 SiLu(swish)와 결과가 거의 비슷하게 나온다.
# 다만, mish가 연산량이 더 많다.
# 때문에 앵간한 데이터에서는 상관없지만 크기가 커지면 SiLu를 써라.