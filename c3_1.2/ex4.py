from sklearn import datasets
import matplotlib.pyplot as plt 
import random 

digit=datasets.load_digits()

plt.figure(figsize=(5,5))
num=random.randrange(1,9)
plt.imshow(digit.images[num],cmap=plt.cm.gray_r,interpolation='nearest')

plt.show()
print(digit.data[num])
print("이 숫자는 ",digit.target[num],"입니다")