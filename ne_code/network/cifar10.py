import tensorflow as tf 
import numpy as np
import time 
import math
max_steps = 5000
batch_size = 1

def variable_with_weight_loss(shape,stddev,wl):
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
        tf.add_to_collection('losses',weight_loss)
    return var

def read_and_decode(filename):      
    filename_queue = tf.train.string_input_producer([filename]) #[filename]   
    reader = tf.TFRecordReader()      
    _, serialized_example = reader.read(filename_queue)           
    features = tf.parse_single_example(serialized_example,features={'label': tf.FixedLenFeature([], tf.int64),'img_raw' : tf.FixedLenFeature([],tf.string),})      
    img = tf.decode_raw(features['img_raw'], tf.uint8)      
    img = tf.reshape(img, [112, 112, 4])    
    img = tf.image.per_image_standardization(img)  
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5      
    label = tf.cast(features['label'], tf.int64)      
    return img, label

images_train, labels_train =read_and_decode('train3.tfrecords')
images_test, labels_test = read_and_decode('test3.tfrecords')
image_batch,label_batch = tf.train.shuffle_batch([images_train,labels_train],batch_size,capacity=50,min_after_dequeue=25)
image_batch1,label_batch1 = tf.train.shuffle_batch([images_test,labels_test],batch_size,capacity=50,min_after_dequeue=25)
#labels = tf.one_hot(label_batch,3,1,0)

image_holder = tf.placeholder(tf.float32,[batch_size,112,112,4])#24243
label_holder = tf.placeholder(tf.int32,[batch_size])

weight1 = variable_with_weight_loss(shape = [5,5,4,64],stddev=5e-2,wl=0.0)
kernel1 = tf.nn.conv2d(image_holder,weight1,[1,1,1,1],padding='SAME')
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
conv1 = tf.nn.relu(tf.nn.bias_add(kernel1,bias1))
pool1 = tf.nn.max_pool(conv1,ksize=[1,3,3,1],strides=[1,2,2,1],padding = 'SAME')
norm1 = tf.nn.lrn(pool1,4,bias=1.0,alpha=0.001/9.0,beta=0.75)

weight2 = variable_with_weight_loss(shape = [5,5,64,64],stddev=5e-2,wl=0.0)
kernel2 = tf.nn.conv2d(norm1,weight2,[1,1,1,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
conv2 = tf.nn.relu(tf.nn.bias_add(kernel2,bias2))
pool2 = tf.nn.max_pool(conv2,ksize=[1,3,3,1],strides=[1,2,2,1],padding = 'SAME')

reshape = tf.reshape(pool2,[batch_size,-1])
dim = reshape.get_shape()[1].value
weight3 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,wl=0.004)
bias3 = tf.Variable(tf.constant(0.1,shape=[384]))
local3 = tf.nn.relu(tf.matmul(reshape,weight3)+bias3)

weight4 =variable_with_weight_loss(shape=[384,192],stddev=0.04,wl=0.004)
bias4 = tf.Variable(tf.constant(0.1,shape=[192]))
local4 = tf.nn.relu(tf.matmul(local3,weight4)+bias4)

weight5 =variable_with_weight_loss(shape=[192,3],stddev=1/192.0,wl=0.0)
bias5 =tf.Variable(tf.constant(0.0,shape=[3]))
logits = tf.add(tf.matmul(local4,weight5),bias5)

def loss(logits,labels):
    labels = tf.cast(labels,tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy,name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)
    return tf.add_n(tf.get_collection('losses'),name='total_loss')

loss = loss(logits,label_holder)
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(logits,label_holder,1)
'''
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
coord = tf.train.Coordinator()
#tf.global_variables_initializer().run()
tf.train.start_queue_runners(coord=coord,sess=sess)
'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.train.start_queue_runners()

for step in range(max_steps):
    start_time = time.time()
    x_batch,y_batch = sess.run([image_batch,label_batch])
    
    _,loss_value = sess.run([train_op,loss],feed_dict = {image_holder: x_batch,label_holder:y_batch})
    duration = time.time() - start_time
    if step % 10==0:
        examples_per_sec = batch_size / duration
        sec_per_batch = float(duration)
        format_str = ('step %d,loss = %.2f(%.1f examples/sec; %.3f sec/batch)')
        print(format_str % (step,loss_value,examples_per_sec,sec_per_batch))
#coord.request_stop()  
#coord.join()  

num_examples = 37
num_iter = int(math.ceil(num_examples / batch_size))
true_count = 0
total_sample_count = num_iter * batch_size
step = 0
while step < num_iter:
    image_batch,label_batch = sess.run([image_batch1, label_batch1])
    #label_batch = tf.one_hot(label_batch,3,1,0)
    predictions = sess.run([top_k_op],feed_dict={image_holder:image_batch,label_holder:label_batch})
    print(predictions)
    true_count += np.sum(predictions)
    step += 1
precision = float(true_count) / float(total_sample_count)
print(true_count, total_sample_count)
print('precision @ 1 = %.3f'%precision)