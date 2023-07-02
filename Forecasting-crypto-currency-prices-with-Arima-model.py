import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import math
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ApiGetData
import ta
from ArimaModel import ArimaModel
from io import StringIO
import sys
import plotly.express as px

#Get API
def getDataApi(sym, timeEnd):
    apiUrl = "https://api.pro.coinbase.com"
    barSize = "86400"

    delta = timedelta(hours=24)
    timeStart = timeEnd - delta * 300

    timeStart = timeStart.isoformat()
    timeEnd = timeEnd.isoformat()

    parameters = {
        "start": timeStart,
        "end": timeEnd,
        "granularity": barSize,
    }

    data = requests.get(f"{apiUrl}/products/{sym}/candles",
                        params=parameters,
                        headers={"content-type": "application/json"})
    return data


def formatData(data):
    df = pd.DataFrame(data.json(),
                      columns=["time", "low", "high", "open", "close", "volume"])
    df['date'] = pd.to_datetime(df['time'], unit='s')
    df.set_index("date", inplace=True)
    df.drop(["time"], axis=1, inplace=True)
    return df


def getAllData(sym):
    timeEnd = datetime.now()
    delta = timedelta(hours=24)
    df_final = pd.DataFrame(columns=["low", "high", "open", "close", "volume"])

    while True:
        db = getDataApi(sym, timeEnd)
        db = formatData(db)
        if len(db.index) != 0:
            df_final = df_final.append(db)
            timeEnd = timeEnd - delta * 300
        else:
            break

    return df_final


def getFinalData(sym, period="DAY"):
    df_origin = getAllData(sym)

    if period == "DAY":
        return df_origin
    else:
        if period == "1WEEK":
            df = df_origin.groupby(pd.Grouper(freq='1W'))
        elif period == "2WEEK":
            df = df_origin.groupby(pd.Grouper(freq='2W'))
        elif period == "MONTH":
            df = df_origin.groupby(pd.Grouper(freq='M'))

        lst_C = []
        for i in df:
            dd = convertData(i)
            lst_C.append(dd)

        final = pd.concat(lst_C)
        final = final[::-1]

        return final


def convertData(tup):
    index = tup[0]
    dataf = tup[1]
    col = dataf.columns
    high = dataf['high'].max()
    low = dataf['low'].min()
    vol = dataf['volume'].sum()
    close = dataf['close'].iloc[-1]
    open = dataf['open'].iloc[0]
    df = pd.DataFrame([low, high, open, close, vol], columns=[index], index=col)
    return df.T


def getListCoins():
    url = "https://api.pro.coinbase.com/currencies"
    response = requests.get(url).json()
    newcoins = []
    dct = {}

    for i in range(len(response)):
        if response[i]['details']['type'] == 'crypto':
            s = response[i]['id'] + "-USD"
            newcoins.append(s)
            n = response[i]['name']
            dct[s] = n

    newcoins.sort()
    tup = tuple(newcoins)

    return tup, dct

#ARIMA Model

class ArimaModel:
    def __init__(self, data, period):
        self.data = data
        self.period = period
        self.result = None
        self.new_model = None
        self.dbReturn = None

    def checkData(self):
        maxday = self.data.index.max()
        minday = self.data.index.min()
        if maxday - minday <= timedelta(days=730):
            warn = "This coin is quite new, the data less than two year, so the model is not reliable enough"
        else:
            warn = "The length of data is enough"
        return warn

    def checkStationarity(self):
        result = adfuller(self.dbReturn)
        if result[1] >= 0.05:
            warn = "P-value > 0.05 => Yield series is non-stationary, the model is not good"
        else:
            warn = "P-value < 0.05 => Yield series is stationary"


        return warn, result[0], result[1]

    def createDataReturn(self):
        self.dbReturn = pd.DataFrame(np.log(self.data['close'] / self.data['close'].shift(1)))
        self.dbReturn = self.dbReturn.fillna(self.dbReturn.head().mean())
        return self.dbReturn

    def displaySummary(self):
        model = auto_arima(self.dbReturn, start_p=1, start_q=1,
                           max_p=10, max_q=10, m=1,
                           start_P=0, seasonal=False,
                           d=0, D=0, trace=True,
                           error_action='ignore',
                           suppress_warnings=True,
                           stepwise=False, max_order=10)

        self.new_model = SARIMAX(self.dbReturn, order=model.order)
        self.result = self.new_model.fit(disp=False)

        return self.result

    def predict(self, delta):
        dic = {"DAY": 1, "1WEEK": 7, "2WEEK": 14, "MONTH": 30}

        latest = self.data.index.max() + timedelta(days=dic.get(self.period))
        date_list = [latest + timedelta(days=x * dic.get(self.period)) for x in range(delta)]

        fc = self.result.get_prediction(start=int(self.new_model.nobs),
                                        end=self.new_model.nobs + delta - 1,
                                        full_reports=True)

        prediction = fc.predicted_mean
        prediction_ci = fc.conf_int()

        prediction = pd.DataFrame(prediction)
        prediction.index = date_list

        prediction_ci = pd.DataFrame(prediction_ci)
        prediction_ci.index = date_list

        prediction.columns = ['predicted_mean']
        lst_mean = self.actualPrice(list(prediction['predicted_mean']))
        lst_upper = self.actualPrice(list(prediction_ci['upper close']))
        lst_lower = self.actualPrice(list(prediction_ci['lower close']))

        date_list_predict = [self.data.index.max() + timedelta(days=x * dic.get(self.period)) for x in range(delta + 1)]

        data_predict = pd.DataFrame({"Price_mean": lst_mean,
                                     "Price_lower": lst_upper,
                                     "Price_upper": lst_lower}, index=date_list_predict)

        return data_predict

    def actualPrice(self, lst):
        l_lastprice = list(self.data['close'].iloc[[0]])
        l_exp = list(math.e ** self.dbReturn['close'].iloc[[0]])

        for i in lst:
            a = math.e ** i
            l_exp.append(a)

        for i in l_exp:
            x = l_lastprice[-1] / i
            l_lastprice.append(x)

        l_lastprice.pop()
        return l_lastprice

#Streamlit App

tup, coinname = ApiGetData.getListCoins()


def main():

    st.title("Predict Token By Using ARIMA")
    st.text("Database load from Coinbase")

    st.sidebar.write("Chose your coin and time")
    coins = st.sidebar.selectbox("Coin", (tup))
    period = st.sidebar.selectbox("Period", ("DAY", "1WEEK", "2WEEK", "MONTH"))

    name = "Coin: " + coinname.get(coins)
    st.subheader(name)
    data = ApiGetData.getFinalData(coins, period)
    st.dataframe(data)

    data["MA20"] = ta.trend.sma_indicator(data['close'], window=20)
    data["MA50"] = ta.trend.sma_indicator(data['close'], window=50)
    data["MA100"] = ta.trend.sma_indicator(data['close'], window=100)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        row_width=[0.2, 0.7])

    # Plot OHLC on 1st row
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['open'],
                                 high=data['high'],
                                 low=data['low'],
                                 close=data['close'], name="OHLC"),
                  row=1, col=1)
    fig.add_trace(go.Line(x=data.index, y=data['MA20'], name="MA20", line=dict(
        color="purple",
        width=1)))
    fig.add_trace(go.Line(x=data.index, y=data['MA50'], name="MA50", line=dict(
        color="yellow",
        width=1.5)))
    fig.add_trace(go.Line(x=data.index, y=data['MA100'], name="MA100", line=dict(
        color="orange",
        width=2)))

    # Bar trace for volumes on 2nd row without legend
    fig.add_trace(go.Bar(x=data.index, y=data['volume'], showlegend=False), row=2, col=1)

    # Do not show OHLC's rangeslider plot
    fig.update(layout_xaxis_rangeslider_visible=False)

    fig.update_layout(
        autosize=False,
        width=780,
        height=540,
        margin=dict(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        )
    )

    st.plotly_chart(fig)

    model = ArimaModel(data, period)

    st.write("Now prepare for the prediction, note that the prediction below uses the Arima model as a reference. You should not apply it to your portfolio and the author will not bear any associated liability.")
    period = st.slider("Chose period you want to predict", 1, 5, 1)
    if st.button("START PREDICT"):
        st.warning(model.checkData())
        model.createDataReturn()
        st.write("Stationality test")
        warn, ADF, p_value = model.checkStationarity()
        s1 = "ADF Statistic: " + str(ADF)
        s2 = "p-value: " + str(p_value)
        st.text(s1)
        st.text(s2)
        st.warning(warn)

        st.markdown("**_Running the auto_arima can take a while._**")

        result = model.displaySummary()

        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        print(result.summary())
        sys.stdout = old_stdout
        st.text(mystdout.getvalue())

        pre = model.predict(period)
        st.write("The data predict")
        st.dataframe(pre)

        fig2 = px.line(data, y="close", x=data.index)
        fig2.add_trace(
            go.Scatter(x=pre.index, y=pre['Price_mean'], line=dict(color="red"), name="forecast"))
        fig2.add_trace(go.Scatter(x=pre.index, y=pre['Price_upper'], line=dict(color="green", dash='dash'), name="upper", ))
        fig2.add_trace(go.Scatter(x=pre.index, y=pre['Price_lower'], line=dict(color="green", dash='dash'), name="lower", ))
        st.plotly_chart(fig2)


if __name__ == '__main__':
    main()