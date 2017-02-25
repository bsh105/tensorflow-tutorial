import numpy as np
import tensorflow as tf
from data_utils import get_CIFAR10_data

# Load the (preprocessed) CIFAR10 data.

data = get_CIFAR10_data()
for k, v in data.iteritems():
  print '%s: ' % k, v.shape

X_train = data['X_train']
y_train = data['y_train']
X_val = data['X_val']
y_val = data['y_val']
X_test = data['X_test']
y_test = data['y_test']

std_image = np.std(X_train, axis=0)
X_train /= std_image
X_val /= std_image
X_test /= std_image

X_test = X_test.reshape(-1, 3072)
y_test_onehot = np.zeros((1000, 10))
y_test_onehot[np.arange(1000), y_test] = 1

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=5e-2)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')



sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 3072])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W_conv1 = weight_variable([5, 5, 3, 64])
b_conv1 = bias_variable([64])

x_image = tf.reshape(x, [-1,32,32,3])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# norm1
norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

W_conv2 = weight_variable([5, 5, 64, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(norm1, W_conv2) + b_conv2)

# norm2
norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

h_pool2 = max_pool_2x2(norm2)

W_fc1 = weight_variable([8 * 8 * 64, 384])
b_fc1 = bias_variable([384])

h_pool2_flat = tf.reshape(h_pool2, [-1, 8*8*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([384, 192])
b_fc2 = bias_variable([192])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([192, 10])
b_fc3 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(15000):

  # Make a minibatch of training data
  num_train = X_train.shape[0]
  batch_mask = np.random.choice(num_train, 50)
  X_batch = X_train[batch_mask].reshape(50, 3072)
  y_indexes = y_train[batch_mask]
  y_batch = np.zeros((50, 10))
  y_batch[np.arange(50), y_indexes] = 1

  if i%100 == 0:
    # Make a minibatch of validation data
    num_val = X_val.shape[0]
    batch_mask_val = np.random.choice(num_val, 50)
    X_batch_val = X_val[batch_mask_val].reshape(50, 3072)
    y_indexes_val = y_val[batch_mask_val]
    y_batch_val = np.zeros((50, 10))
    y_batch_val[np.arange(50), y_indexes_val] = 1

    train_accuracy = accuracy.eval(feed_dict={
        x:X_batch.astype(np.float32), y_: y_batch.astype(np.float32), keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
    val_accuracy = accuracy.eval(feed_dict={
        x:X_batch_val.astype(np.float32), y_: y_batch_val.astype(np.float32), keep_prob: 1.0})
    print("step %d, validation accuracy %g"%(i, val_accuracy))


  _, loss_val = sess.run([train_step, cross_entropy],
                    feed_dict={x: X_batch.astype(np.float32), y_: y_batch.astype(np.float32), keep_prob: 0.5})




print("test accuracy %g"%accuracy.eval(feed_dict={
    x: X_test, y_: y_test_onehot, keep_prob: 1.0}))
