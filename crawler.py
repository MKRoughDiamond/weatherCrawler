from urllib.request import urlopen
from bs4 import BeautifulSoup

temp = []
hum = []
start = 1960
end = 2018

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


f = open('data.csv', 'w')
for i in range(len(hum)):
    f.write(str(temp[i])+','+str(hum[i])+'\n')
f.close()
print('done')
