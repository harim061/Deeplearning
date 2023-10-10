import tensorflow as tf


x=[[0.0,0.0],[0.0,1.0],[1.0,0.0],[1.0,1.0]]
y=[[-1],[1],[1],[1]]


w=tf.Variable([[1.0],[1.0]])
b=tf.Variable(-0.5)


s=tf.add(tf.matmul(x,w),b)
o=tf.sign(s)

print(o)