from sklearn import datasets
from sklearn import svm

digit = datasets.load_digits()

s=svm.SVC(gamma=0.1,C=10)
s.fit(digit.data,digit.target)

new_d=[digit.data[0],digit.data[1],digit.data[2]]
res=s.predict(new_d)
print("에측값은", res)
print("참값은",digit.target[0],digit.target[1],digit.target[2])

res=s.predict(digit.data)
correct=[i for i in range(len(res)) if res[i]==digit.target[i]]

accuracy=len(correct)/len(res)
print("화소 특정을 사용했을 때 정확률=",accuracy*100,"%")