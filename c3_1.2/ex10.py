from sklearn import datasets,svm 
from sklearn import tree


d=datasets.load_iris()

s=svm.SVC(gamma=0.1)
s.fit(d.data,d.target)

res=s.predict(d.data)
correct=[i for i in range(len(res)) if res[i]==d.target[i]]
accuracy=len(correct)/len(res)
print("svm : ", accuracy*100)

s2=tree.DecisionTreeClassifier()
s2.fit(d.data,d.target)

res2=s2.predict(d.data)
correct=[i for i in range(len(res2)) if res2[i]==d.target[i]]
accuracy=len(correct)/len(res2)
print("tree : ", accuracy*100)