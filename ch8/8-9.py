import os
import numpy as np
from scipy.spatial import distance


fname='./glove.6B/glove.6B.100d.txt'
f=open(fname,encoding='utf8')

for line in f: 
    print(type(line))
    print(line)
    break


dictionary={}
for line in f:
    li=line.split()
    word=li[0]
    vector=np.asarray(li[1:],dtype='float32')
    dictionary[word]=vector


def find_closest_words(vector):
    return sorted(dictionary.keys(), key=lambda w: distance.euclidean(dictionary[w],vector))


print(find_closest_words(dictionary['movie'])[:5])
print(find_closest_words(dictionary['school'])[:5])
print(find_closest_words(dictionary['oak'])[:5])


print(find_closest_words(dictionary["seoul"]-dictionary["korea"]+dictionary["spain"])[:5])
print(find_closest_words(dictionary["animal"]-dictionary["lion"]+dictionary["oak"])[:5])
print(find_closest_words(dictionary["queen"]-dictionary["king"]+dictionary["actress"])[:5])

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


tsne=TSNE(n_components=2,random_state=0)
words=list(dictionary.keys())
vectors=[dictionary[word] for word in words]
p2=tsne.fit_transform(vectors[:100])
plt.scatter(p2[:,0],p2[:,1])

for label,x,y in zip(words,p2[:,0],p2[:,1]):
    plt.annotate(label,xy=(x,y))