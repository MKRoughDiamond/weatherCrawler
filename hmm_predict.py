import numpy as np
from matplotlib import cm, pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os
import warnings

#warning구문 지우기
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning) 

#작업 폴더 설정
work_dir = os.path.dirname(os.path.realpath(__file__))
filename= work_dir+"/data.csv"

#초기 설정값
max_temp=-10000
min_temp=10000
max_hum=-10000
max_wind=-10000
n_components = 10

#입력
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

#데이터 normalize
def preprocessing(data):
    data_ = [[(item[0]-min_temp)/(max_temp-min_temp),item[1]/max_hum,item[2]/max_wind] for item in data]
    diff = np.diff(data_,axis=0)
    return np.concatenate((diff,data_[:-1]),axis=1),data_


data = csv_reader()
X,data = preprocessing(data)

#데이터 학습
model = GaussianHMM(n_components=n_components,covariance_type="diag",n_iter=25).fit(X[:-4300])
print("hmm learning done")

#학습된 cluster의 mean, std를 얻음
mean = [item[0:3] for item in model.means_]
std = [[np.sqrt(item[i][i]) for i in range(0,3)] for item in model.covars_]

print("mean of deltas")
print(mean)
print("std of deltas")
print(std)

#기온 예측
predict=[]
curr = X[:-4300]
for i in range(100):
    state = model.predict(curr)
    delta = np.asarray([np.random.normal(mean[state[-1]][j],std[state[-1]][j]) for j in range(3)])
    curr=curr[1:]
    curr=np.concatenate((curr,[np.array(np.append(delta,curr[-1][3:6]+delta))]))
    predict.append([item for item in curr[-1][3:6]])
    
plt.plot([item[0] for item in predict],'r',alpha=0.8)
plt.plot([item[3] for item in X[-4300:-4200]],'b',alpha=0.4)
plt.xlabel("days")
plt.ylabel("Temperature(normalized)")
plt.title("Temperature prediction with HMM")
plt.show()
