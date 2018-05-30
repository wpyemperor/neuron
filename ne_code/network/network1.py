#第1层，卷积层  
#这里参数W_conv1,b_conv1,x_image的含义参见博客http://blog.csdn.net/xbcreal/article/details/71811506  
#初始化W为[5,5,1,32]的tensor，表示卷积核大小为5*5，第一层网络的输入和输出神经元个数分别为1和32  
W_conv1 = weight_variable([5,5,1,32])  
#初始化b为[32],即输出大小,因为有多少个卷积核就会由多少个偏置项b  
b_conv1 = bias_variable([32])  
  
#把输入x(二维tensor,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]  
#-1表示自动推测这个维度的size  
x_image = tf.reshape(x, [-1,28,28,1])  
  
#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling  
#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,1]  
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  
h_pool1 = max_pool_2x2(h_conv1)  
  
#第2层，卷积层  
#卷积核大小依然是5*5，这层的输入和输出神经元个数为32和64  
W_conv2 = weight_variable([5,5,32,64])  
b_conv2 = weight_variable([64])  
  
#h_pool2即为第二层网络输出，shape为[batch,7,7,1],因为是在14x14的h_pool1上卷积再2x2的最大值池化的,所以变为了7x7  
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
h_pool2 = max_pool_2x2(h_conv2)  
  
#第3层, 全连接层  
#这层是拥有1024个神经元的全连接层  
#W的第1维size为7*7*64，7*7是h_pool2输出的size，64是第2层输出神经元个数  
W_fc1 = weight_variable([7*7*64, 1024])  
b_fc1 = bias_variable([1024])  
  
#计算前需要把第2层的输出reshape成[batch, 7*7*64]的张量  
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)  
  
#Dropout层  
#为了减少过拟合，在输出层前加入dropout  
keep_prob = tf.placeholder("float")  
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  
  
#输出层  
#最后，添加一个softmax层  
#可以理解为另一个全连接层，只不过输出时使用softmax将网络输出值转换成了概率  
W_fc2 = weight_variable([1024, 10])  
b_fc2 = bias_variable([10])  
  
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  
  
#预测值和真实值之间的交叉墒  
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))  
  
#train op, 使用ADAM优化器来做梯度下降。学习率为0.0001  
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  
  
#评估模型，tf.argmax能给出某个tensor对象在某一维上数据最大值的索引。  
#因为标签是由0,1组成了one-hot vector，返回的索引就是数值为1的位置  
correct_predict = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))  
  
#计算正确预测项的比例，因为tf.equal返回的是布尔值，  
#使用tf.cast把布尔值转换成浮点数，然后用tf.reduce_mean求平均值  
accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))  
  
#初始化变量  
sess.run(tf.global_variables_initializer())  
  
#开始训练模型，循环20000次，每次随机从训练集中抓取50幅图像  
for i in range(20000):  
    batch = mnist.train.next_batch(50)  
    if i%100 == 0:  
        #每100次输出一次日志  
        train_accuracy = accuracy.eval(feed_dict={  
            x:batch[0], y_:batch[1], keep_prob:1.0})  
        print "step %d, training accuracy %g" % (i, train_accuracy)  
  
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})  
  
print "test accuracy %g" % accuracy.eval(feed_dict={  
x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})  