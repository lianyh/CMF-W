import tensorflow as tf
import numpy as np
import os
import scipy.io
import math
import sys
from sklearn.metrics import roc_auc_score


# define Frobenius norm square
def frob(z, k1, k2):
    vec_i = tf.reshape(z, [-1])
    return tf.reduce_sum(tf.multiply(vec_i, vec_i))

def train(max_steps, tol, xa_data, xi_data, xg_data, xs_data, xt_data, xu_data, y_train, y_test, du1, du2, iter, train_size, lambda1):
    kdti, da = xa_data.shape
    ndti, di = xi_data.shape
    n, dg = xg_data.shape
    _, ds = xs_data.shape
    tf.set_random_seed(1)

    sess = tf.InteractiveSession()

    # Input placeholders
    with tf.name_scope("input"):
        xa = tf.placeholder(tf.float32, shape=(None, da), name='xa-input')
        xi = tf.placeholder(tf.float32, shape=(None, di), name='xi-input')
        xg = tf.placeholder(tf.float32, shape=(None, dg), name='xg-input')
        xs = tf.placeholder(tf.float32, shape=(None, ds), name='xs-input')
        xt = tf.placeholder(tf.float32, shape=(None, dg), name='xt-input')
        xu = tf.placeholder(tf.float32, shape=(None, ds), name='xu-input')
        ytest = tf.placeholder(tf.float32, shape=(None, 3), name='ytest-input')
        keep_prob = tf.placeholder(tf.float32)

    # initialize all factors by svd, you can choose other way to initialize all the variables
    with tf.name_scope('svd'):
        udti_svd1, _, vdti_svd1 = np.linalg.svd(xi_data, full_matrices=False)
        udti_svd2, _, vdti_svd2 = np.linalg.svd(udti_svd1, full_matrices=False)

        ut1_svd1, _, vt1_svd1 = np.linalg.svd(xg_data, full_matrices=False)
        ut1_svd2, _, vt1_svd2 = np.linalg.svd(ut1_svd1, full_matrices=False)

        vi = tf.cast(tf.Variable(vt1_svd1[0: du2, :]), tf.float32)

        u1dti = udti_svd2[:, 0:du2]

    [a0,b0]=udti_svd1.shape
    name = str(du1) + str(du2) + str(lambda1) + str(iter)

    w0 = tf.get_variable("w0_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
    w1 = tf.get_variable("w1_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
    w2 = tf.get_variable("w2_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
    w3 = tf.get_variable("w3_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
    w4 = tf.get_variable("w4_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())
    w5 = tf.get_variable("w5_" + name, shape=(a0, b0), initializer=tf.contrib.layers.xavier_initializer())

    u1i = tf.Variable(tf.cast(u1dti, tf.float32))

    bias = tf.Variable(tf.constant(0.1, shape=[1,1]))

    with tf.name_scope('output'):
        y_conf =  tf.matmul(tf.matmul(u1i,tf.transpose(u1i)),w0)

    row_train = y_train[:,0]
    col_train = y_train[:,1]

    tf_row_train = tf.cast(row_train, tf.int32)
    tf_col_train = tf.cast(col_train, tf.int32)

    loss = frob(xa - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w0), tf_row_train, tf_col_train) +\
           frob(xi - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w1), tf_row_train, tf_col_train) +\
           frob(xg - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w2), tf_row_train, tf_col_train) +\
           frob(xs - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w3), tf_row_train, tf_col_train) +\
           frob(xt - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w4), tf_row_train, tf_col_train) +\
           frob(xu - tf.matmul((tf.matmul(u1i, tf.transpose(u1i))), w5), tf_row_train, tf_col_train) +\
           lambda1*frob_orig(u1i) 

    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    tf.global_variables_initializer().run()

    def feed_dict(training, prob_rate):
        xas = xa_data
        xis = xi_data
        xgs = xg_data
        xss = xs_data
        xst = xt_data
        xsu = xu_data
        ys = y_test
        return {xa: xas, xi: xis, xg: xgs, xs:xss, xt:xst, xu:xsu, ytest: ys, keep_prob: prob_rate}

    funval = []
    _, loss_iter = sess.run([train_step, loss], feed_dict=feed_dict(True, 0.8))
    funval.append(loss_iter)

    for i in range(max_steps):
        _, loss_iter = sess.run([train_step, loss], feed_dict=feed_dict(True, 0.8))
        funval.append(loss_iter)

        if abs(funval[i+1] - funval[i]) < tol:
            break
        if math.isnan(loss_iter):
            break

    pred_conf = sess.run([y_conf], feed_dict=feed_dict(False, 1.0))
    label_test = y_test[:,2]
    rows=np.array(y_test[:,0],dtype=np.intp)
    cols=np.array(y_test[:,1],dtype=np.intp)

    pred_conf_reshape1 = np.reshape(np.array(pred_conf), (train_size, train_size))
    pred_conf_reshape = pred_conf_reshape1[rows,cols]
    test_auc = roc_auc_score(label_test, pred_conf_reshape)
    
    rows_train =np.array(y_train[:,0],dtype=np.intp)
    cols_train =np.array(y_train[:,1],dtype=np.intp)
    label_train=y_train[:,2]

    pred_train_reshape = pred_conf_reshape1[rows_train,cols_train]
    train_auc = roc_auc_score(label_train, pred_train_reshape)

    w0,w1,w2,w3,w4,w5, u1i, bias, vi = sess.run([w0,w1,w2,w3,w4,w5,u1i,bias,vi], feed_dict=feed_dict(True, 1.0))
    
    c = tf.Print([train_auc,test_auc], [train_auc,test_auc], message="The train_auc and test_auc: ")
    print(sess.run(c))

    d = tf.Print([du2,iter], [du2,iter], message="The du2 and iter: ")
    print(sess.run(d))

    sess.close()
    return {'funval': funval, 'y_test_conf':pred_conf_reshape, 'train_auc': train_auc, 'test_auc': test_auc, 'w0': w0,'w1': w1,'w2': w2,'w3': w3,'w4': w4,'w5': w5, 'u1i': u1i, 'bias': bias, 'vi':vi}


def load_data(iter):
	x0=np.loadtxt(sys.argv[1])
	x1=np.loadtxt(sys.argv[2])
	x2=np.loadtxt(sys.argv[3])
	x3=np.loadtxt(sys.argv[4])
	x4=np.loadtxt(sys.argv[5])
	x5=np.loadtxt(sys.argv[6])

	y_train=np.loadtxt(sys.argv[7])
	y_test=np.loadtxt(sys.argv[8])
	train_size = sys.argv[9]

	y_train=y_train.reshape(int(y_train.size/3),3)
	y_test=y_test.reshape(int(y_test.size/3),3)

	return {'x0':x0, 'x1': x1, 'x2': x2,'x3':x3, 'x4': x4,'x5':x5, 'y_train':y_train, 'y_test': y_test, 'train_size': train_size}

def main():
    max_iter = 200000
    tol = 1e-7
    directory = "resultF2"
    if not os.path.exists(directory):
        os.makedirs(directory)
    du1=50
    directory1 = directory + "/du1" + str(du1)
    if not os.path.exists(directory1):
        os.makedirs(directory1)
    for du2 in [10, 20, 30, 40, 50,60]:
        directory2 = directory1 + "/du2" + str(du2)
        if not os.path.exists(directory2):
            os.makedirs(directory2)
        aucfile = open(directory2 + "/auc.txt", 'w')
        for lambdax in[1e-4]:
            lambda1 =lambdax 
            auc = []
            train_auc = []
            # iteration number
            for it in [1, 2, 3, 4, 5]:
                data = load_data(it)
                x0 = data['x0']
                x1 = data['x1']
                x2 = data['x2']
                x3 = data['x3']
                x4 = data['x4']
                x5 = data['x5']

                y_train = data['y_train']	
                y_test = data['y_test']
                train_size = data['train_size']
                # train the network
                result = train(max_iter, tol, x0, x1, x2, x3, x4, x5, y_train, y_test,du1, du2, it, train_size,lambda1)
                auc.append(result['test_auc'])
                print("iter:" + str(it) + ",train_auc:"+str(result['train_auc'])+",test auc:"+ str(result['test_auc']))
                train_auc.append(result['train_auc'])
			

            auc_mean = np.mean(auc)
            train_auc_mean = np.mean(train_auc)
            auc_std = np.std(auc)
            print("lambdax=" +str(lambdax) + ",train_auc_mean="+str(train_auc_mean)+",test_auc_mean=" + str(auc_mean))
            aucfile.write("lambda1 %s:auc_mean %s auc_std %s\n"
						  % (lambda1,auc_mean, auc_std))
        aucfile.close()
if __name__ == "__main__":
    main()