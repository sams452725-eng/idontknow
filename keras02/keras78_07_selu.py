import numpy as np
import matplotlib.pyplot as plt

x=np.arange(-5,5,0.1)

# SELU 고정 상수(논문 제안값)
alpha=1.6732632423543772
lmbda=1.0507009873554805        # 원 논문에서는 lambda가 맞는데 현재는 아래 함수에서 lambda가 중복 됨으로 현재는 변수를 따로 만들었다.

# def selu(x,alpha,lmbda):
#     return lmbda*((x>0)*x+(x<=0)*(alpha*(np.exp(x)-1)))

selu=lambda x, alpha, lmbda: lmbda*((x>0)*x+(x<=0)*(alpha*(np.exp(x)-1)))

y=selu(x, alpha, lmbda)

plt.plot(x,y)
plt.grid()
plt.show()
