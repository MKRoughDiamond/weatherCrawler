from urllib.request import urlopen
from bs4 import BeautifulSoup

temp = []
hum = []
wind = []
start = 1960
end = 2018
month=[31,28,31,30,31,30,31,31,30,31,30,31]

'''TEMPERATURE'''

for i in range(start,end):
    print('collect avg. temperature on {}'.format(i))
    r = urlopen('http://www.weather.go.kr/weather/climate/past_table.jsp?stn=108&yy='+str(i)+
                '&obs=07').read().decode('euc-kr', 'ignore')
    soup = BeautifulSoup(r, "html5lib")
    tableParser = soup.find_all('tbody')[1].find_all('tr')[:-1]
    tempList = [[] for _ in range(12)]
    for j in tableParser:
        tempParser = j.find_all('td')
        for k in range(1,13):
            try:
                tempList[k-1].append(float(tempParser[k].string))
            except:
                pass         
    for j in range(12):
        f = open('./temp/data{}-{}.csv'.format(i,j+1), 'w')
        for k in tempList[j]:
            f.write(str(k)+'\n')
        f.close()
        temp = temp + tempList[j]

        
'''HUMIDITY'''

for i in range(start,end):
    print('collect avg. humidity on {}'.format(i))
    r = urlopen('http://www.weather.go.kr/weather/climate/past_table.jsp?stn=108&yy='+str(i)+
                '&obs=12').read().decode('euc-kr', 'ignore')
    soup = BeautifulSoup(r, "html5lib")
    tableParser = soup.find_all('tbody')[1].find_all('tr')[:-1]
    humList = [[] for _ in range(12)]
    for j in tableParser:
        humParser = j.find_all('td')
        for k in range(1,13):
            try:
                humList[k-1].append(float(humParser[k].string))
            except:
                pass 
    for j in range(12):
        f = open('./hum/data{}-{}.csv'.format(i,j+1), 'w')
        for k in humList[j]:
            f.write(str(k)+'\n')
        f.close()
        hum = hum + humList[j]

        
'''WIND'''

for i in range(start,end):
    print('collect avg. wind on {}'.format(i))
    r = urlopen('http://www.weather.go.kr/weather/climate/past_table.jsp?stn=108&yy='+str(i)+
                '&obs=06').read().decode('euc-kr', 'ignore')
    soup = BeautifulSoup(r, "html5lib")
    tableParser = soup.find_all('tbody')[1].find_all('tr')[:-1]
    windList = [[] for _ in range(12)]
    for j in tableParser:
        windParser = j.find_all('td')
        for k in range(1,13):
            try:
                windList[k-1].append(float(windParser[k].string))
            except:
                if k==2:
                    if i%4==0 and (i%400==0 or i%100!=0) and len(windList[k-1])<=28:
                        windList[k-1].append(windList[k-1][-1])
                    elif len(windList[k-1])<month[k-1]:
                        windList[k-1].append(windList[k-1][-1])
                else:
                    if len(windList[k-1])<month[k-1]:
                        windList[k-1].append(windList[k-1][-1])
    for j in range(12):
        f = open('./wind/data{}-{}.csv'.format(i,j+1), 'w')
        for k in windList[j]:
            f.write(str(k)+'\n')
        f.close()
        wind = wind + windList[j]
        

f = open('data.csv', 'w')
for i in range(len(hum)):
    f.write(str(temp[i])+','+str(hum[i])+','+str(wind[i])+'\n')
f.close()
print('done')
