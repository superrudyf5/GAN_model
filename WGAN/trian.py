import numpy as np
import dataIO
import os
import tensorflow as tf


iterations = 10000
z_size = 200
batch_size = 50
n_critic = 5
g_lr  = 5e-5
d_lr  = 5e-5
leak_value = 0.2
cube_len = 64

volumes = dataIO.getAll(cube_len=cube_len)
volumes = volumes[..., np.newaxis].astype(np.float)
out_directory = './out/'
model_directory = './model/'

weights,biases = {},{}
def generator(z,batch_size = batch_size,phase_train=True,reuse = False):
    strides = [1,2,2,2,1]
    with tf.variable_scope('gen', reuse=reuse):
        z = tf.reshape(z,(batch_size,1,1,1,z_size))
        g_1 = tf.nn.conv3d_transpose(z,weights['wg1'],(batch_size,4,4,4,512), strides=[1,1,1,1,1],padding='VALID')
        g_1 = tf.contrib.layers.batch_norm(g_1, is_training=phase_train)
        g_1 = tf.nn.relu(g_1)

        g_2 = tf.nn.conv3d_transpose(g_1,weights['wg2'], output_shape=(batch_size,8,8,8,256), strides=strides,padding='SAME')
        g_2 = tf.contrib.layers.batch_norm(g_2, is_training=phase_train)
        g_2 = tf.nn.relu(g_2)

        g_3 = tf.nn.conv3d_transpose(g_2,weights['wg3'], output_shape=(batch_size,16,16,16,128), strides=strides,padding='SAME')
        g_3 = tf.contrib.layers.batch_norm(g_3, is_training=phase_train)
        g_3 = tf.nn.relu(g_3)

        g_4 = tf.nn.conv3d_transpose(g_3,weights['wg4'], output_shape=(batch_size,32,32,32,64), strides=strides,padding='SAME')
        g_4 = tf.contrib.layers.batch_norm(g_4, is_training=phase_train)
        g_4 = tf.nn.relu(g_4)

        g_5 = tf.nn.conv3d_transpose(g_4,weights['wg5'], output_shape=(batch_size,64,64,64,1), strides=strides,padding='SAME')
        g_5 = tf.nn.sigmoid(g_5)
    print(g_1, 'g1')
    print(g_2, 'g2')
    print(g_3, 'g3')
    print(g_4, 'g4')
    print(g_5, 'g5')
    return(g_5)
def discriminator(inputs,phase_train=True, reuse=False):
    strides = [1,2,2,2,1]
    with tf.variable_scope('dis', reuse=reuse):
        d_1 = tf.nn.conv3d(inputs,weights['wd1'],strides=strides,padding='SAME')
        d_1 = tf.contrib.layers.layer_norm(d_1)
        d_1 = tf.nn.leaky_relu(d_1,leak_value)

        d_2 = tf.nn.conv3d(d_1,weights['wd2'],strides=strides,padding='SAME')
        d_2 = tf.contrib.layers.layer_norm(d_2)
        d_2 = tf.nn.leaky_relu(d_2,leak_value)

        d_3 = tf.nn.conv3d(d_2,weights['wd3'],strides=strides,padding='SAME')
        d_3 = tf.contrib.layers.layer_norm(d_3)
        d_3 = tf.nn.leaky_relu(d_3,leak_value)

        d_4 = tf.nn.conv3d(d_3,weights['wd4'],strides = strides,padding='SAME')
        d_4 = tf.contrib.layers.layer_norm(d_4)
        d_4 = tf.nn.leaky_relu(d_4,leak_value)

        d_5 = tf.nn.conv3d(d_4,weights['wd5'],strides=[1,1,1,1,1],padding='SAME')

    print(d_1, 'd1')
    print(d_2, 'd2')
    print(d_3, 'd3')
    print(d_4, 'd4')
    print(d_5, 'd5')
    return d_5
def initialiseWeights():
    global weights
    xavier_init = tf.contrib.layers.xavier_initializer()

    weights['wg1'] = tf.get_variable('wg1',shape=[4,4,4,512,z_size],initializer=xavier_init)
    weights['wg2'] = tf.get_variable('wg2',shape=[4,4,4,256,512],initializer=xavier_init)
    weights['wg3'] = tf.get_variable('wg3',shape=[4,4,4,128,256],initializer=xavier_init)
    weights['wg4'] = tf.get_variable('wg4',shape=[4,4,4,64,128],initializer=xavier_init)
    weights['wg5'] = tf.get_variable('wg5',shape=[4,4,4,1,64],initializer=xavier_init)

    weights['wd1'] = tf.get_variable('wd1',shape=[4,4,4,1,64],initializer=xavier_init)
    weights['wd2'] = tf.get_variable('wd2',shape=[4,4,4,64,128],initializer=xavier_init)
    weights['wd3'] = tf.get_variable('wd3',shape=[4,4,4,128,256],initializer=xavier_init)
    weights['wd4'] = tf.get_variable('wd4',shape=[4,4,4,256,512],initializer=xavier_init)
    weights['wd5'] = tf.get_variable('wd5',shape=[4,4,4,512,1],initializer=xavier_init)
def trainGAN():
    weights = initialiseWeights()
    z = tf.placeholder(shape=[batch_size,z_size],dtype=tf.float32)
    X = tf.placeholder(shape=[batch_size,cube_len,cube_len,cube_len,1], dtype=tf.float32)

    G_sample = generator(z, phase_train=True, reuse=False)
    D_real = discriminator(X, phase_train=True, reuse=False)
    D_fake = discriminator(G_sample, phase_train=True, reuse=True)

    D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
    G_loss = -tf.reduce_mean(D_fake)

    theta_G = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wg','gen'])]
    theta_D = [var for var in tf.trainable_variables() if any(x in var.name for x in ['wd','dis'])]

    D_solver = tf.train.RMSPropOptimizer(learning_rate=d_lr).minimize(-D_loss,var_list=theta_D)
    G_solver = tf.train.RMSPropOptimizer(learning_rate=g_lr).minimize(G_loss,var_list=theta_G)

    clip_D = [p.assign(tf.clip_by_value(p,-0.01,0.01)) for p in theta_D]

    saver = tf.train.Saver(max_to_keep=50)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    index_in_epoch = 0

    for it in range(iterations):
        for _ in range(5):
            start = index_in_epoch
            index_in_epoch += batch_size
            if index_in_epoch > len(volumes):
                start = 0
                index_in_epoch = batch_size
            end = index_in_epoch
            X_mb = volumes[start:end]

            z_sample = np.random.normal(size=[batch_size, z_size]).astype(np.float32)
            _,D_loss_curr,_ = sess.run([D_solver,D_loss,clip_D],feed_dict={X:X_mb,z:z_sample})

        z_sample = np.random.normal(size=[batch_size, z_size]).astype(np.float32)
        _,G_loss_curr = sess.run([G_solver,G_loss],feed_dict={z:z_sample})

        if it % 100==0:
            print('Iter:{}; D loss:{:4}; G loss:{:4}'.format(it,D_loss,G_loss))
        if it % 500 == 0:
            z_sample = np.random.normal(size=[batch_size, z_size]).astype(np.float32)
            samples = sess.run(G_sample,feed_dict={z:z_sample})
            if not os.path.exists(out_directory):
                os.makedirs(out_directory)
            np.save(out_directory + '/' + str(iterations) + '_model.npy', samples)

            if not os.path.exists(model_directory):
                os.makedirs(model_directory)
            saver.save(sess,save_path=model_directory + '/' +str(iterations) + '.cptk')

if __name__ == '__main__':
    trainGAN()