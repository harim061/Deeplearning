from sklearn.linear_model import Perceptron

# (1)데이터 분포를 나타내는 훈련 집합
X=[[0,0,0],[0,1,0],[1,0,0],[1,1,0],[1,1,1],[1,0,1],[0,0,1],[0,1,1]]
y=[1,1,1,1,1,1,-1,-1]

# (2)데이터를 인식하는 퍼셉트론
p=Perceptron()
p.fit(X,y)

# (3) 실행 결과
print("매개변수:", p.coef_,p.intercept_)
print("예측:", p.predict(X))
print("정확률:", p.score(X,y)*100, "%")

