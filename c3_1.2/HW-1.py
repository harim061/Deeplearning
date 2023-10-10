from sklearn import datasets,svm
import matplotlib.pyplot as plt

d=datasets.load_iris()
print(d.DESCR)

for i in range(0,len(d.data)):
    print(i+1, d.data[i],d.target[i])


s=svm.SVC(gamma=0.1,C=10)
s.fit(d.data,d.target)

new_d=[[6.4,3.2,6.0,2.5],[7.1,3.1,4.7,1.35]]

res=s.predict(new_d)
print("새로운 2개 샘플의 부류는",res)