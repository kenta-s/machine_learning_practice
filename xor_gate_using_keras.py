import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

model = Sequential()

model.add(Dense(input_dim=2, units=2))
model.add(Activation('sigmoid'))

model.add(Dense(units=1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))

model.fit(X, Y, epochs=4000, batch_size=4)

classes = model.predict_classes(X, batch_size=4)
prob = model.predict_proba(X, batch_size=4)

# x = tf.placeholder(tf.float32, shape=[None, 2])
# t = tf.placeholder(tf.float32, shape=[None, 1])
# 
# W = tf.Variable(tf.truncated_normal([2, 2]))
# b = tf.Variable(tf.zeros([2]))
# h = tf.nn.sigmoid(tf.matmul(x, W) + b)
# 
# V = tf.Variable(tf.truncated_normal([2, 1]))
# c = tf.Variable(tf.zeros([1]))
# y = tf.nn.sigmoid(tf.matmul(h, V) + c)
# 
# cross_entropy = tf.reduce_sum(t* tf.log(y) + (1 - t) * tf.log(1 - y))
# 
# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
# correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t)
# 
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
# 
# for epoch in range(4000):
#     sess.run(train_step, feed_dict={
#         x: X,
#         t: Y
#     })
#     if epoch % 1000 == 0:
#         print('epoch:', epoch)
# 
# classified = correct_prediction.eval(session=sess, feed_dict={
#     x: X,
#     t: Y
# })
# prob = y.eval(session=sess, feed_dict={
#     x: X,
#     t: Y
# })

print('classified:')
print(Y == classes)
print()
print('output probability:')
print(prob)
