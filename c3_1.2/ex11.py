from sklearn import datasets,svm 
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

d=datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(d.data,d.target,train_size=0.7)

s=svm.SVC(gamma=0.001)
acc=cross_val_score(s,x_train,y_train,cv=5)

print(acc)
print("svm : 평균 -",acc.mean()*100)

s2=tree.DecisionTreeClassifier()
acc2=cross_val_score(s2,x_train,y_train,cv=5)

print(acc2)
print("tree : 평균 -",acc2.mean()*100)