import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist
import matplotlib.pyplot as plt
import numpy as np

X,Y,X_test,Y_test = mnist.load_data()
img_dim = 784
z_dim = 200
total_sample = len(X)

def generate(x,reuse=tf.AUTO_REUSE):
    with tf.variable_scope('Generate',reuse=reuse):
        x = tflearn.fully_connected(x,256,activation='relu')
        x = tflearn.fully_connected(x,img_dim,activation='sigmoid')
        return x
def discrimination(x,reuse = tf.AUTO_REUSE):
    with tf.variable_scope('Discriminator',reuse=reuse):
        x = tflearn.fully_connected(x,256,activation='relu')
        x = tflearn.fully_connected(x,1,activation='sigmoid')
        return x
gen_input = tflearn.input_data(shape=[None,z_dim],name='input_noise')
disc_input = tflearn.input_data(shape=[None,784],name = 'disc_input')

gen_sample = generate(gen_input)
disc_real = discrimination(disc_input)
disc_fake = discrimination(gen_sample)

disc_loss = -tf.reduce_mean(tf.log(disc_real)) + tf.log(1. -disc_fake)
gen_loss = -tf.reduce_mean(tf.log(disc_fake))

gen_vars = tflearn.get_layer_variables_by_name('Generate')
gen_model = tflearn.regression(gen_sample,placeholder=None,optimizer='adam',loss=gen_loss,trainable_vars=gen_vars,
                               batch_size=64,name='target_gen',op_name='GEN')
disc_vars = tflearn.get_layer_variables_by_scope('Discriminator')
disc_model = tflearn.regression(disc_real,placeholder=None,optimizer='adam',loss=disc_loss,trainable_vars=disc_vars,
                                batch_size=64,name='target_disc',op_name='DISC')
gan = tflearn.DNN(gen_model)

z = np.random.uniform(-1.,1.,[total_sample,z_dim])
gan.fit(X_inputs = {gen_input:z,disc_input:X},Y_targets=None,n_epoch =200)

f,a = plt.subplot(2,10,figsize=(10,4))
for i in range(10):
    for j in range(2):
        z = np.random.uniform(-1.,1.,size=[1,z_dim])

        temp = [[temp,temp, temp]for temp in list(gan.predict([z])[0])]
        print(temp)
        a[j][i].imshow(np.reshape(temp,(28,28,3)))

f.show()
plt.show()
