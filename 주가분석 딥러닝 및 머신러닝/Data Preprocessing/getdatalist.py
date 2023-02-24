from re import I
from bs4 import BeautifulSoup
import requests
import sys
import math
from FinanceDataReader import *
import pandas as pd
import ta
from sklearn.preprocessing import MinMaxScaler
import numpy as np 
class HaltedItemcodeError(Exception):
    def __init__(self,itemcode):
        self.itemcode=itemcode
    def __str__(self):
        return f"{self.itemcode}: 거래 정지가 된 종목입니다."      
class InvalidItemcodeError(Exception):
    def __init__(self,itemcode):
        self.itemcode=itemcode
    def __str__(self):
        return f"{self.itemcode}: 종목 번호가 잘못되었습니다. 종목 번호는 반드시 6자리의 숫자여야 합니다."
class WrongUSCodeError(Exception):
    def __init__(self,itemcode):
        self.itemcode=itemcode
    def __str__(self):
        return f"{self.itemcode}: 잘못된 종합지수 검색 코드입니다. 미국 종합지수 코드는 <DJI,IXIC,US500,VIX> 넷 중 하나만 사용할 수 있습니다."
    
    
def getcompanylist(address):
    company_df=pd.read_csv(address)
    #여기 절대 경로를 사용자에 맞추어 바꾸어 주세요.
    companylist=company_df["종목코드"].to_list()
    for i in range(len(companylist)):
        companylist[i]=str(companylist[i]).rjust(6,"0")#padding 함수 사용
    return companylist


def getTotaldf(dom_df,cur_df,USI_df,tick):
    Total_df=pd.merge(dom_df,cur_df,on="Date")
    Total_df=pd.merge(Total_df,USI_df,on="Date")
    Total_df.tail(10)
    if tick:
        yd_data=Total_df['Close']
        yc_data=Total_df["C-Close"]
        yi_data=Total_df["I-Close"]
        yd_data.loc[-1]=Total_df["Close"].iloc[0]-(Total_df["Close"].iloc[0]*Total_df["Change"].iloc[0])/(1-Total_df["Change"].iloc[0])
        yc_data.loc[-1]=Total_df["C-Close"].iloc[0]-(Total_df["C-Close"].iloc[0]*Total_df["C-Change"].iloc[0])/(1-Total_df["C-Change"].iloc[0])
        yi_data.loc[-1]=Total_df["I-Close"].iloc[0]-(Total_df["I-Close"].iloc[0]*Total_df["I-Change"].iloc[0])/(1-Total_df["I-Change"].iloc[0])
        #첫 번째 변화량은 연산이 불가능하므로 주어진 데이터를 통하여 역산하여 추측함.
        yd_data.index=yd_data.index+1
        yc_data.index=yc_data.index+1
        yi_data.index=yi_data.index+1
        yd_data=yd_data.sort_index()
        yc_data=yc_data.sort_index()
        yi_data=yi_data.sort_index()
        yd_data=yd_data.drop(index=len(yd_data)-1,axis=0)
        yc_data=yc_data.drop(index=len(yc_data)-1,axis=0)
        yi_data=yi_data.drop(index=len(yi_data)-1,axis=0)
        Total_df=Total_df.assign(Yd_close=yd_data)
        Total_df=Total_df.assign(Yc_close=yc_data)
        Total_df=Total_df.assign(Yi_close=yi_data)
        Total_df["Difference"]=Total_df["Close"]-Total_df["Yd_close"]
        Total_df["C-Difference"]=Total_df["C-Close"]-Total_df["Yc_close"]
        Total_df["I-Difference"]=Total_df["I-Close"]-Total_df["Yi_close"]
        Total_df["Change"]=Total_df["Difference"]/Total_df["Yd_close"]*100
        Total_df["C-Change"]=Total_df["C-Difference"]/Total_df["Yc_close"]*100
        Total_df["I-Change"]=Total_df["I-Difference"]/Total_df["Yi_close"]*100
        Total_df=Total_df.drop(labels=["Yd_close","Yc_close","Yi_close"],axis=1)
        Total_df = Total_df.replace([np.inf, -np.inf], np.nan) # replace 메서드로 np.inf를 None(np.nan)으로 변경
        Total_df=Total_df.dropna()#np.inf가 들어간 부분(비정상적인 급등이나 급락이 있는 부분)을 삭제함.
        Total_df["Difference"]=Total_df["Difference"].astype(np.int32)
        Total_df=Total_df[Total_df["Volume"]!=0]#거래량이 0인 부분도 삭제함.
    return Total_df

def gethalteddata():
    pagenum=1
    while True:
        halteditemcode=[]
        maxpagenum=0
        url=f'https://www.kokstock.com/stock/halt.asp?page={pagenum}'
        header={"User-Agent" : "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.75 Safari/537.36"}
        res=requests.get(url,headers=header)
        soup=BeautifulSoup(res.text,features="lxml")
        pagedata=soup.find('b',{"id":"searchTotal"})
        maxpagenum=math.ceil(int(pagedata.text)/20)#불러오기 최소화를 위하여 이 형태로 최대 페이지를 지정함.
        data=soup.findAll('span',{'title':"더블클릭시 클립보드에 복사됩니다"})
        for items in data:
            halteditemcode.append(items.text)
        if (pagenum<maxpagenum):
            pagenum+=1
        else:
            break
    return halteditemcode

def idxvalidatycheck(halteditemcode,ValidUSidx,Domidx=None,USidx=None):
#값의 무결성 검사-국내주식 종목
    if Domidx is not None:
        for numbers in Domidx:
            try:
                _=int(numbers)#번호에 문자가 포함되어 있지 않은지 확인
                if len(Domidx)!=6:
                    raise InvalidItemcodeError(Domidx)
            except ValueError:#내부에 문자가 단 한개라도 포함되었을 경우 오류메시지 출력
                raise InvalidItemcodeError(Domidx)
            if Domidx in halteditemcode:#Domidx(국내주식 종목번호)에 거래정지된 주식이 포함되어있을 경우
                raise HaltedItemcodeError(Domidx)
    #값의 무결성 검사-미국주가지수 지표
    if USidx is not None:
        if USidx not in ValidUSidx:
            raise WrongUSCodeError(USidx)

start=1992#시작 일자
end=0#끝 일자
USidx="IXIC"#미국주식 지수 고유값
Curidx="USD/KRW"#환율:기준 환율(1)/변환 환율(여러 값이 나올 수 있음.)
ValidUSidx=["VIX","IXIC","DJI","US500"]
ma = [5,20,60,120]
companyaddress="C:\programming_practice\python\python_programming\project\listed_companies.csv"
if(end==0):
    end=None
else:
    end=str(end)
start=str(start)
halteditemlist=gethalteddata()
Domidxlist=getcompanylist(companyaddress)

for Domidx in Domidxlist:
    print(f"now processing: {Domidx}")
    try:
        idxvalidatycheck(halteditemlist,ValidUSidx,Domidx,USidx)#무결성 체크
        dom_df=DataReader(Domidx,start,end)#데이터가 없을 경우 그 전 데이터를 가져옴.
        dom_df.info()#Null 체크
        dom_df=dom_df.reset_index()
        df=DataReader(USidx,start,end).reset_index()
        USI_df=pd.DataFrame(columns=["Date","I-Close","I-Change"])
        USI_df["Date"]=df["Date"]
        USI_df["I-Close"]=df["Close"]
        USI_df["I-Change"]=df["Change"]
        read_df=DataReader(Curidx,start,end).reset_index()
        cur_df=pd.DataFrame(columns=["Date","C-Close","C-Change"])
        cur_df[["Date","C-Close","C-Change"]]=read_df[["Date","Close","Change"]]
        #여기에 Total_df를 구하는 함수가 있어야 함.
        Total_df=getTotaldf(dom_df,cur_df,USI_df,True)
        H, L, C, V =Total_df['High'], Total_df['Low'], Total_df['Close'], Total_df['Volume']
        sdf = pd.DataFrame()
        sdf['MFI'] = ta.volume.money_flow_index(high=H, low=L, close=C, volume=V, fillna=True)
        sdf['ADI'] = ta.volume.acc_dist_index(high=H, low=L, close=C, volume=V, fillna=True)
        sdf['OBV'] = ta.volume.on_balance_volume(close=C, volume=V, fillna=True)
        sdf['MACD'] = ta.trend.macd(close=C, fillna=True)
        sdf['CCI'] = ta.trend.cci(high=H, low=L, close=C, fillna=True)
        sdf['RSI'] = ta.momentum.rsi(close=C, fillna=True)
        sdf['BOL_H'] = ta.volatility.bollinger_hband(close=C, fillna=True)
        sdf['BOL_L'] = ta.volatility.bollinger_lband(close=C, fillna=True)
        sdf=sdf.fillna(0)
        new_total=pd.concat([Total_df,sdf],axis=1)
        new_total=new_total.set_index("Date",drop=True).sort_index()
        for days in ma:
            new_total['ma_'+str(days)] = new_total['Close'].rolling(window = days).mean().round()
        scaler = MinMaxScaler()
        scale_price = ["Close", "I-Close", "C-Close"]
        p_scaled = scaler.fit_transform(new_total[scale_price])
        p_scaled = pd.DataFrame(p_scaled)
        p_scaled.columns = scale_price
        p_scaled.index = pd.to_datetime(new_total.index)
        scale_indicator = ["MACD", "BOL_H", "BOL_L","ma_5", "ma_20", "ma_60", "ma_120"]
        i_scaled = scaler.fit_transform(new_total[scale_indicator])
        i_scaled = pd.DataFrame(i_scaled)
        i_scaled.columns = scale_indicator
        i_scaled.index = pd.to_datetime(new_total.index)
        p_scaled.to_csv(f'project/companydatas/Pricescaled_{Domidx}_ref_{USidx}.csv')
        i_scaled.to_csv(f"project/companydatas/supportindexscaled_{Domidx}_ref_{USidx}.csv")
        Total_df.to_csv(f"project/companydatas/Totaldata_{Domidx}_ref_{USidx}.csv")
    except:#거래 정지된 데이터로 인하여 오류가 발생 시 그 데이터를 넘김.
        continue