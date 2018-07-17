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
        x_in = tf.concat(x_data,x_generate)
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
    x_vector = tf.placeholder(shape=[batch_size, NUM_POLYGONS, 9, 1], dtype=tf.float32)

    net_g_train = generator(z_vector, phase_train=True, reuse=False)

    d_output_x = discriminator(x_vector, phase_train=True, reuse=False)
    d_output_x = tf.maximum(tf.minimum(d_output_x, 0.99), 0.01)
    summary_d_x_hist = tf.summary.histogram("d_prob_x", d_output_x)

    d_output_z = discriminator(net_g_train, phase_train=True, reuse=True)
    d_output_z = tf.maximum(tf.minimum(d_output_z, 0.99), 0.01)
    summary_d_z_hist = tf.summary.histogram("d_prob_z", d_output_z)

    # Compute the discriminator accuracy
    n_p_x = tf.reduce_sum(tf.cast(d_output_x > 0.5, tf.int32))
    n_p_z = tf.reduce_sum(tf.cast(d_output_z <= 0.5, tf.int32))
    d_acc = tf.divide(n_p_x + n_p_z, 2 * batch_size)


    # Compute the discriminator and generator loss
    d_loss = -tf.reduce_mean(tf.log(d_output_x) + tf.log(1 - d_output_z))
    g_loss = -tf.reduce_mean(tf.log(d_output_z))

    summary_d_loss = tf.summary.scalar("d_loss", d_loss)
    summary_g_loss = tf.summary.scalar("g_loss", g_loss)
    summary_n_p_z = tf.summary.scalar("n_p_z", n_p_z)
    summary_n_p_x = tf.summary.scalar("n_p_x", n_p_x)
    summary_d_acc = tf.summary.scalar("d_acc", d_acc)

    net_g_test = generator(z_vector, phase_train=False, reuse=True)

    para_g = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg', 'bg', 'gen'])]
    para_d = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wd', 'bd', 'dis'])]

    # only update the weights for the discriminator network
    optimizer_op_d = tf.train.AdamOptimizer(learning_rate=d_lr, beta1=beta).minimize(d_loss, var_list=para_d)
    # only update the weights for the generator network
    optimizer_op_g = tf.train.AdamOptimizer(learning_rate=g_lr, beta1=beta).minimize(g_loss, var_list=para_g)

    saver = tf.train.Saver(max_to_keep=50)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        z_sample = np.random.normal(0, 0.33, size=[batch_size, z_size]).astype(np.float32)

        for epoch in range(n_epochs):
            for start, end in zip(
                    range(0, len(polygonBatch), batch_size),
                    range(batch_size, len(polygonBatch), batch_size)):

                next_polygon = polygonBatch[start:end].reshape(batch_size,NUM_POLYGONS,9,1)

                d_summary_merge = tf.summary.merge([summary_d_loss,
                                                    summary_d_x_hist,
                                                    summary_d_z_hist,
                                                    summary_n_p_x,
                                                    summary_n_p_z,
                                                    summary_d_acc])

                summary_d, discriminator_loss = sess.run([d_summary_merge, d_loss],
                                                         feed_dict={z_vector: z_sample, x_vector: next_polygon})

                summary_g, generator_loss = sess.run([summary_g_loss, g_loss], feed_dict={z_vector: z_sample})
                # d_output_z, d_output_x = sess.run([d_acc, n_p_x, n_p_z],
                #                                 feed_dict={z_vector: z_sample, x_vector: next_polygon})

                d_accuracy, n_x, n_z = sess.run([d_acc, n_p_x, n_p_z], feed_dict={z_vector: z_sample, x_vector: next_polygon})
                print('-epoch{}--n_p_x:{}--n_p_z:{}--'.format(
                     epoch,n_x, n_z))

                if d_accuracy < d_thresh:
                    sess.run([optimizer_op_d], feed_dict={z_vector: z_sample, x_vector: next_polygon})
                    print('Discriminator Training ', "epoch: ", epoch, ', d_loss:', discriminator_loss, 'g_loss:',
                          generator_loss, "d_acc: ", d_accuracy)

                sess.run([optimizer_op_g], feed_dict={z_vector: z_sample})
                print('Generator Training ', "epoch: ", epoch, ', d_loss:', discriminator_loss, 'g_loss:',
                      generator_loss, "d_acc: ", d_accuracy)

                # output generated chairs
                if epoch % 100 == 10:
                # if epoch ==0:
                    g_model = sess.run(net_g_test, feed_dict={z_vector: z_sample})
                    if not os.path.exists(train_sample_directory):
                        os.makedirs(train_sample_directory)
                    print('-----========-------=====-----------=====-------------')
                    getMeshFromMatrix(g_model.reshape(batch_size,NUM_POLYGONS,9),train_sample_directory,epoch)
                    # g_model.dump(train_sample_directory + '/' + str(epoch))

                if epoch % 100 == 10:
                # if epoch==0:
                    if not os.path.exists(model_directory):
                        os.makedirs(model_directory)
                    saver.save(sess, save_path=model_directory + '/' + str(epoch) + '.cptk')

if __name__ == '__main__':
    # is_dummy =

    trainGAN()





