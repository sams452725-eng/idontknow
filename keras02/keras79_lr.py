x=10
y=10
w=0.001
lr=0.999
epochs=1000

for i,e in enumerate(range(epochs)):
    hypothesis=x*w
    loss=(y-hypothesis)**2      # 해당 식은 mse다. 바이어스와 시그마는 우선 신경안쓴다.

    print(i+1, 'Loss:', round(loss,4), '\tPredict:', round(hypothesis,4))

    up_predict=x*(w+lr)
    up_loss=(y-up_predict)**2

    down_predict=x*(w-lr)
    down_loss=(y-down_predict)**2

    if (up_loss>down_loss):
        w=w-lr
    else:
        w=w+lr


