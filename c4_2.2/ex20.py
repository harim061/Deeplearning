from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
import numpy as np

# (1)데이터 분포를 나타내는 훈련 집합
X=[[0,0,0],[0,1,0],[1,0,0],[1,1,0],[1,1,1],[1,0,1],[0,0,1],[0,1,1]]
y=[1,1,1,0,1,1,0,1]

p=Perceptron()
p.fit(X,y)

# (2) 75% 인식함 퍼셉트론은 선형이라서 100%로 해당 데이터 인식 불가능
print("매개변수:", p.coef_,p.intercept_)
print("예측:", p.predict(X))
print("정확률:", p.score(X,y)*100, "%")

# (3) 다층 퍼셉트론을 제시
mlp = MLPClassifier(hidden_layer_sizes=(3), learning_rate_init=0.001, max_iter=5000,activation='tanh',  solver='adam', random_state=1)
mlp.fit(X, y)
res=mlp.predict(X)


conf=np.zeros((2,2))
for i in range(len(res)):
    conf[res[i]][y[i]]+=1
print(conf)

no_correct=0
for i in range(2):
    no_correct+=conf[i][i]
accuracy=no_correct/len(res)

print("다층 퍼셉트론 : ",accuracy*100,"%")