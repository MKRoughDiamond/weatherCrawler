import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

#작업폴더 설정, 기본값 세팅
work_dir = os.path.dirname(os.path.realpath(__file__))
filename = work_dir + "/data.csv"
step_size = 5
batch_size = 10
hyperparameters=3
loss_l=[]

#lstm 세팅
lstm_size=3
lstm_units=3
forget_bias = 1.0

#변수 세팅(normalize관련)
max_temp=-10000
min_temp=10000
max_hum=-10000
max_wind=-10000

#데이터 입력
def csv_reader():
    with open(filename) as inf:
        global max_temp,max_hum,max_wind,min_temp
        l = []
        for line in inf:
            line = line[:-1]
            temp,hum,wind = line.split(',')
            l.append([float(temp),float(hum),float(wind)])
            if(max_temp<float(temp)):
                max_temp=float(temp)
            if(max_hum<float(hum)):
                max_hum=float(hum)
            if(max_wind<float(wind)):
                max_wind=float(wind)
            if(min_temp>float(temp)):
                min_temp=float(temp)
        return l

#데이터 전처리
def normalize(l):
    global max_temp,max_hum,max_wind,min_temp
    return [(l[0]-min_temp)/(max_temp-min_temp),l[1]/max_hum,l[2]/max_wind]

def preprocessing(data):
    data_x=[]
    data_y=[]
    data_norm=[]
    for item in data:
        data_norm.append(normalize(item))
    for i in range(len(data)-step_size):
        data_x.append(data_norm[i:i+step_size])
        data_y.append(data_norm[i+step_size])
    return data_x,data_y

#lstm
def lstm_cells(size):
    return [tf.contrib.rnn.BasicLSTMCell(lstm_units,forget_bias,state_is_tuple=True) for _ in range(size)]

#rnn 구조의 선언 (LSTM cell을 붙인 MultiRNN cell + DNN)
def rnn(x):
    lstm = tf.contrib.rnn.MultiRNNCell(lstm_cells(lstm_size),state_is_tuple=True)
    x_, state_ = tf.nn.static_rnn(lstm,x,dtype=tf.float32)
    w1 = tf.layers.dense(x_[-1],units=60)
    w2 = tf.layers.dense(w1,units=60,activation=tf.nn.sigmoid)
    output = tf.layers.dense(w2,units=hyperparameters)
    return output

#작업을 위한 graph 설정, loss는 SE의 max를 최소화하는 방향으로
g = tf.Graph()
with g.as_default():
    X = tf.placeholder(tf.float32, [None,step_size,hyperparameters])
    Y = tf.placeholder(tf.float32, [None,hyperparameters])
    
    X_ = tf.unstack(X,num=step_size,axis=1)
    y_ = rnn(X_)

    loss = tf.reduce_max(tf.square(Y-y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train = optimizer.minimize(loss)

#tf session
with tf.Session(graph = g) as sess:
    data = csv_reader()
    data_size = len(data)
    data_x,data_y=preprocessing(data)
    loss_l = []

    train_size = data_size*4//5
    predict_size = data_size-train_size

    sess.run(tf.global_variables_initializer())
    total_batchs = (train_size-step_size)//batch_size
    for i in range(total_batchs-1):
        batch_x=data_x[i*batch_size:(i+1)*batch_size]
        batch_y=data_y[i*batch_size:(i+1)*batch_size]
        sess.run(train,feed_dict = {X:batch_x,Y:batch_y})
        loss_norm = sess.run(loss,feed_dict={X:batch_x,Y:batch_y})
        loss_l.append(loss_norm)
        if i % 50==0:
            print("===============================")
            print("batch : {}".format(i+1))
            print("loss : {}".format(loss_norm))

    predict_temp=[]
    for i in range(predict_size-step_size):
        predict_temp.append(sess.run(y_,feed_dict={X:[data_x[i+train_size]]})[0])
    t = range(1,len(predict_temp)+1)
    
    #graph     
    plt.plot(loss_l,color='r',linewidth=0.5)
    plt.xlabel('days')
    plt.ylabel('loss')
    plt.title('RNN loss')
    plt.savefig("loss.png")
    plt.clf()
    
    plt.plot(t,[item[0] for item in predict_temp],color='r',linewidth=0.5,alpha=0.8)
    plt.plot(t,[item[0] for item in data_y[train_size:]],color='b',linewidth=0.5,alpha=0.6)
    plt.xlabel('days')
    plt.ylabel('Temp.(normalized)')
    plt.title('Temperature prediction(RNN)')
    plt.savefig("prediction_temp.png")
    plt.clf()
    
    plt.plot(t,[item[1] for item in predict_temp],color='r',linewidth=0.5,alpha=0.8)
    plt.plot(t,[item[1] for item in data_y[train_size:]],color='b',linewidth=0.5,alpha=0.6)
    plt.xlabel('days')
    plt.ylabel('Hum.(normalized)')
    plt.title('Humidity prediction(RNN)')
    plt.savefig("prediction_hum.png")
    plt.clf()
    
    plt.plot(t,[item[2] for item in predict_temp],color='r',linewidth=0.5,alpha=0.8)
    plt.plot(t,[item[2] for item in data_y[train_size:]],color='b',linewidth=0.5,alpha=0.6)
    plt.xlabel('days')
    plt.ylabel('Wind(normalized)')
    plt.title('Wind prediction(RNN)')
    plt.savefig("prediction_wind.png")
    plt.close()
    
