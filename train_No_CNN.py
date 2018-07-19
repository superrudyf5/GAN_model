from model import *
from load import loadData
import numpy as np
from utils import lrelu
from utils import getMeshFromMatrix
import os

n_epochs = 30000
learning_rate = 0.008
batch_size = 27
z_size = 200
h1_size = 150
h2_size = 300

g_lr  = 0.008
d_lr  = 0.000001
beta  = 0.5
d_thresh   = 0.8
if os.path.exists('polygonData.npy'):
    polygonBatch = np.load('polygonData.npy')
else:
    polygonBatch = loadData(1)

NUM_POLYGONS = 576
img_size = NUM_POLYGONS * 9
model_directory = './mlxFile1/'
train_sample_directory = './train_no_CNN_sample/'

weights,biases = {},{}

def generator(z,batch_size = batch_size,phase_train=True,reuse = False):
    print('------------z shape {}------------------'.format(z.shape))
    # strides = [1,4,1,1]
    with tf.variable_scope("gen", reuse=reuse):
        wg_1 = tf.Variable(tf.truncated_normal([z_size,h1_size],stddev=0.1),name="g_w1",dtype=tf.float32)
        b_1 = tf.Variable(tf.zeros([h1_size]),name='g_b1',dtype=tf.float32)
        h_1 = tf.nn.relu(tf.matmul(z,wg_1)+b_1)

        wg_2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
        b_2 = tf.Variable(tf.zeros([h2_size]), name='g_b2', dtype=tf.float32)
        h_2 = tf.nn.relu(tf.matmul(h_1, wg_2) + b_2)

        wg_3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
        b_3 = tf.Variable(tf.zeros([img_size]), name='g_b3', dtype=tf.float32)
        h_3 = tf.nn.relu(tf.matmul(h_2, wg_3) + b_3)

        x_generate = tf.nn.tanh(h_3)
        g_params = [wg_1,b_1,wg_2,b_2,wg_3,b_3]

        #
        # g_5 = tf.nn.conv2d_transpose(g_4, weights['wg5'], (batch_size, 576, 9, 1), strides=[1,4,1,1], padding="SAME")
        # g_5 = tf.nn.bias_add(g_5, biases['bg5'])


    return x_generate,g_params

def discriminator(x_data,x_generate, phase_train=True, reuse=False,keep_prob = 0.8):
    # print('------------inputs in Dis {}------------------'.format(inputs.shape))
    # strides = [1, 3, 3, 1]
    with tf.variable_scope("dis", reuse=reuse):
        x_in = tf.concat([x_data,x_generate],0)
        wd_1 = tf.Variable(tf.truncated_normal([img_size,h2_size],stddev = 0.1),name='d_w1',dtype=tf.float32)
        b_1 = tf.Variable(tf.zeros([h2_size]),name='d_b1',dtype=tf.float32)
        h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in,wd_1)+b_1),keep_prob)

        wd_2 = tf.Variable(tf.truncated_normal([h2_size,h1_size],stddev=0.1),name='d_w2',dtype=tf.float32)
        b_2 = tf.Variable(tf.zeros([h1_size]),name='d_b2',dtype=tf.float32)
        h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1,wd_2)+b_2),keep_prob)

        wd_3 = tf.Variable(tf.truncated_normal([h1_size,1],stddev=0.1),name='d_w3',dtype=tf.float32)
        b_3 = tf.Variable(tf.zeros([1]),name='d_b3',dtype=tf.float32)
        h3 = tf.matmul(h2,wd_3) +b_3

        y_data = tf.nn.sigmoid(tf.slice(h3,[0,0],[batch_size,-1],name=None))
        y_generated = tf.nn.sigmoid(tf.slice(h3,[batch_size,0],[-1,-1],name=None))
        d_params = [wd_1,b_1,wd_2,b_2,wd_3,b_3]
    return y_data,y_generated,d_params




def trainGAN():

    z_vector = tf.placeholder(shape=[batch_size, z_size], dtype=tf.float32)
    x_data = tf.placeholder(shape=[batch_size, NUM_POLYGONS*9], dtype=tf.float32)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    x_generated,g_params = generator(z_vector, phase_train=True, reuse=False)

    y_data, y_generated, d_params = discriminator(x_data,x_generated, phase_train=True, reuse=False)

    d_loss = -(tf.log(y_data) + tf.log(1-y_generated))
    g_loss = -tf.log(y_generated)

    optimizer = tf.train.AdadeltaOptimizer(0.0001)
    d_trainer = optimizer.minimize(d_loss,var_list=d_params)
    g_trainer = optimizer.minimize(g_loss,var_list=g_params)
    init = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=50)

    with tf.Session() as sess:
        sess.run(init)
        z_sample = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)
        count = 0
        for epoch in range(n_epochs):
            for start, end in zip(
                    range(0, len(polygonBatch), batch_size),
                    range(batch_size, len(polygonBatch), batch_size)):

                next_polygon = polygonBatch[start:end].reshape(batch_size,NUM_POLYGONS*9)
                z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
                sess.run(d_trainer,
                         feed_dict={x_data: next_polygon, z_vector: z_value, keep_prob: np.sum(0.7).astype(np.float32)})

                if count % 1 == 0:
                    sess.run(g_trainer,
                             feed_dict={x_data: next_polygon, z_vector: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
                    print('Generator Training ', "epoch: ", epoch, ', d_loss:', d_trainer, 'g_loss:',
                        g_trainer)
                else:
                    print('Discriminator Training ', "epoch: ", epoch, ', d_loss:', d_trainer, 'g_loss:',
                          g_trainer)
                if count %100 == 10:
                    x_gen_val = sess.run(x_generated, feed_dict={z_vector: z_sample})
                    if not os.path.exists(train_sample_directory):
                        os.makedirs(train_sample_directory)
                    print('-----========-------=====-----------=====-------------')
                    getMeshFromMatrix(x_gen_val.reshape(batch_size, NUM_POLYGONS, 9), train_sample_directory, epoch)

                if epoch % 100 == 10:
                # if epoch==0:
                    if not os.path.exists(model_directory):
                        os.makedirs(model_directory)
                    saver.save(sess, save_path=model_directory + '/' + str(epoch) + '.cptk')

                count += 1


if __name__ == '__main__':
    # is_dummy =

    trainGAN()





