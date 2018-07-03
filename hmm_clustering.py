import numpy as np
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os
import warnings

#warning구문 삭제
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 
#작업폴더 설정
work_dir = os.path.dirname(os.path.realpath(__file__))
filename= work_dir+"/data.csv"

#초기값
max_temp=-10000
min_temp=10000
max_hum=-10000
max_wind=-10000

#그래프 관련
colors=["chartreuse","red","blue","yellow","purple"]
y_labels=["Temp.(normalized)","Hum.(normalized)","Wind(normalized)"]

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
def preprocessing(data):
    data_ = [[(item[0]-min_temp)/(max_temp-min_temp),item[1]/max_hum,item[2]/max_wind] for item in data]
    diff = np.diff(data_,axis=0)
    return np.concatenate((diff,data_[1:]),axis=1),data_

data = csv_reader()
X,data = preprocessing(data)
#데이터 학습
model = GaussianHMM(n_components=5,covariance_type="full",n_iter=25,algorithm="viterbi").fit(X)
#state predict
hidden_states = model.predict(X)

#그래프 표현
fig, axs=plt.subplots(3,sharex=True,sharey=False)
plt.suptitle("hmm clustering")
for a,ax in enumerate(axs):
    for i in range(5):
        mask = hidden_states==i
        lx=[]
        ly=[]
        for j in range(len(mask)*4//5,len(mask)):
            if mask[j]:
                lx.append(X[j][a])
                ly.append(X[j][a+3])
        ax.scatter(lx,ly,color=colors[i],marker='.',alpha=1)
    ax.set_ylabel(y_labels[a])
plt.xlabel("delta")
plt.show()
