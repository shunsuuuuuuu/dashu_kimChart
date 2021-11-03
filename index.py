# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 23:14:35 2021

@author: chibi
"""

#! pip install yahoo_fin
from googletrans import Translator
import yahoo_fin.stock_info as si
# import fix_yahoo_finance as yf
import yfinance as yf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
import mplfinance as mpf
from datetime import timedelta
from plotly.subplots import make_subplots
import sys
import plotly.express as px
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
import numpy as np   
import dash  
import dash_core_components as dcc 
import dash_html_components as html  
import plotly.graph_objs as go 
import json 
import requests
import os
import glob
import yahoo_fin.stock_info as si
from datetime import datetime as dt
from matplotlib import dates
# %matplotlib inline
from dash.dependencies import Input, Output, State
import plotly.io as pio

# In[]:SBI取り扱い銘柄を抽出
'''
import requests
from bs4 import BeautifulSoup
import re
urlName = "https://search.sbisec.co.jp/v2/popwin/info/stock/pop6040_usequity_list.html"
url = requests.get(urlName)
page_soup = BeautifulSoup(url.content, "html.parser")
stock_list = page_soup.select('th', class_='md-l-table-01 md-l-utl-mt10')
stock_list=[x.text for x in stock_list]

stock_table = page_soup.select('td', class_='md-l-table-01 md-l-utl-mt10')
stock_table=[x.text for x in stock_table]

stock_list_ = pd.DataFrame(stock_list[4:4010])
stock_business = stock_table[31:12050]
stock_business = pd.DataFrame(stock_business[1::3])
df = pd.concat([stock_list_,stock_business],axis=1)
df.columns = ['symbol','Business Content']
df.to_csv('SBI銘柄リスト.csv',index=False)
'''
# In[]:

stock_table = pd.read_csv('SBI銘柄リスト.csv')
stock_list = list(pd.read_csv('SBI銘柄リスト.csv')['symbol'])
stock_business = list(pd.read_csv('SBI銘柄リスト.csv')['Business Content'])

stock_dict_list=[]
for stk in stock_list:
    stock_dict=dict(zip(['label','value'],[stk,stk]))
    stock_dict_list.append(stock_dict)


# In[]:


# EPS情報の取得
def getStockInfo(ticker):
    ticker_analysts_info = si.get_analysts_info(ticker)
    # DATA_ReveEst = ticker_analysts_info["Revenue Estimate"]
    # DATA_EarnEst = ticker_analysts_info["Earnings Estimate"]
    DATA_EarnHist = ticker_analysts_info["Earnings History"]
    DATA_EarnHist.index = DATA_EarnHist["Earnings History"]
    DATA_EarnHist = DATA_EarnHist.drop('Earnings History',axis=1)
    DATA_EarnHist = DATA_EarnHist.transpose()
    
    ticker_financials = si.get_financials(ticker)
    DATA_income_state_Q = ticker_financials['quarterly_income_statement']
    DATA_income_state_Y = ticker_financials['yearly_income_statement']
        #当期純利益、営業利益、総売上、総営業費用、収益のコスト
    DATA_income_state_Q = DATA_income_state_Q.loc[["netIncome",'operatingIncome',
                                        "totalRevenue",'totalOperatingExpenses',
                                        'costOfRevenue']]
    DATA_income_state_Y = DATA_income_state_Y.loc[["netIncome",'operatingIncome',
                                        "totalRevenue",'totalOperatingExpenses',
                                        'costOfRevenue']]
    DATA_income_state_Q = DATA_income_state_Q[DATA_income_state_Q.columns[::-1]]
    DATA_income_state_Y = DATA_income_state_Y[DATA_income_state_Y.columns[::-1]]
    
    DATA_income_state_Q.index=["当期純利益","営業利益","総売上","総営業費用","収益のコスト"]
    DATA_income_state_Y.index=["当期純利益","営業利益","総売上","総営業費用","収益のコスト"]
    
    DATA_income_state_Q = DATA_income_state_Q.transpose()
    DATA_income_state_Y = DATA_income_state_Y.transpose()
    
    # DATA_income_state_Q['営業利益率'] = DATA_income_state_Q['営業利益']/DATA_income_state_Q['総売上']*100
    # DATA_income_state_Y['営業利益率'] = DATA_income_state_Y['営業利益']/DATA_income_state_Y['総売上']*100
    # DATA_sale_Q=DATA_income_state_Q.loc["総売上"]
    # DATA_sale_Y=DATA_income_state_Y.loc["総売上"]
    # DATA_profit_Q=DATA_income_state_Q.loc["当期純利益"]
    # DATA_profit_Y=DATA_income_state_Y.loc["当期純利益"]
    # DATA_Q = pd.concat([DATA_sale_Q,DATA_profit_Q],axis=1)
    return DATA_EarnHist,DATA_income_state_Q
    
    # width=0.3
    # left = np.arange(len(DATA_sale_Q))
    # labels = (DATA_sale_Q.index).astype(str)
    # plt.bar(left,DATA_sale_Q,width=width)
    # plt.bar(left+width,DATA_profit_Q,width=width)
    # plt.xticks(left + width/2, labels)
    # plt.show()
    
    # width=0.3
    # left = np.arange(len(DATA_EarnHist))
    # labels = (DATA_EarnHist.index)#.astype(str)
    # plt.bar(left,DATA_EarnHist['EPS Est.'].astype(float),width=width)
    # plt.bar(left+width,DATA_EarnHist['EPS Actual'].astype(float),width=width)
    # plt.xticks(left + width/2, labels)
    # plt.show()
    
def mk_plotData(slct_stock):
    print(slct_stock[0])
    plotdata=pd.DataFrame()
    prisedata=pd.DataFrame()
    for ticker in slct_stock[0]:
        ticker_get_data = si.get_data(ticker)
        prisedata = pd.concat([prisedata,ticker_get_data['close']],axis=1)
        ticker_get_data = ticker_get_data.sort_index(axis='index',ascending=False)
        ticker_get_data['diff'] = ticker_get_data['close'] - ticker_get_data['open']
        ticker_get_data['up-down_day'] = ticker_get_data['close']/ticker_get_data['open']
        ticker_get_data['up-down_week']=ticker_get_data['close']/ticker_get_data['close'].shift(-5)
        ticker_get_data['up-down_month'] = ticker_get_data['close']/ticker_get_data['close'].shift(-20)
        print(len(ticker_get_data))
        if len(ticker_get_data)<240:
            ticker_get_data['up-down_year'] = ticker_get_data['close']/ticker_get_data['close'].shift(-len(ticker_get_data)+1)
        else:
            ticker_get_data['up-down_year'] = ticker_get_data['close']/ticker_get_data['close'].shift(-240)
        ticker_get_data['日変動率'] = (ticker_get_data['up-down_day']-1)*100
        ticker_get_data['週変動率'] = (ticker_get_data['up-down_week']-1)*100
        ticker_get_data['月変動率'] = (ticker_get_data['up-down_month']-1)*100
        ticker_get_data['年変動率'] = (ticker_get_data['up-down_year']-1)*100
        ticker_get_data = ticker_get_data.iloc[0]
        plotdata=pd.concat([plotdata,ticker_get_data],axis=1)
    # plotdata.index = plotdata["code"]
    plotdata=plotdata.transpose()
    prisedata.columns =  slct_stock[0]
    return plotdata

def macd(df):
    FastEMA_period = 12  # 短期EMAの期間
    SlowEMA_period = 26  # 長期EMAの期間
    SignalSMA_period = 9  # SMAを取る期間
    df["MACD"] = df["close"].ewm(span=FastEMA_period).mean() - df["close"].ewm(span=SlowEMA_period).mean()
    df["Signal"] = df["MACD"].rolling(SignalSMA_period).mean()
    return df


def rsi(df):
    # 前日との差分を計算
    df_diff = df["close"].diff(1)
    # 計算用のDataFrameを定義
    df_up, df_down = df_diff.copy(), df_diff.copy()
    # df_upはマイナス値を0に変換
    # df_downはプラス値を0に変換して正負反転
    df_up[df_up < 0] = 0
    df_down[df_down > 0] = 0
    df_down = df_down * -1
    # 期間14でそれぞれの平均を算出
    df_up_sma14 = df_up.rolling(window=14, center=False).mean()
    df_down_sma14 = df_down.rolling(window=14, center=False).mean()
    # RSIを算出
    df["RSI"] = 100.0 * (df_up_sma14 / (df_up_sma14 + df_down_sma14))
    return df

def mk_Teckdata(ticker,period_num):
    my_share = share.Share(ticker)
    symbol_data = None
     
    try:
        symbol_data = my_share.get_historical(
            share.PERIOD_TYPE_MONTH, period_num,
            share.FREQUENCY_TYPE_DAY, 1)
    except YahooFinanceError as e:
        print(e.message)
        sys.exit(1)
    
    df = pd.DataFrame(symbol_data)
    df["datetime"] = pd.to_datetime(df.timestamp, unit="ms")
    
    
    d_all = pd.date_range(start=df['datetime'].iloc[0],end=df['datetime'].iloc[-1])
    d_obs = [d.strftime("%Y-%m-%d") for d in df['datetime']]
    d_breaks = [d for d in d_all.strftime("%Y-%m-%d").tolist() if not d in d_obs]
    
    # SMAを計算
    df["SMA5"] = df["close"].rolling(window=5).mean()
    df["SMA25"] = df["close"].rolling(window=25).mean()
     
    # MACDを計算する
    df = macd(df)
    # RSIを算出
    df = rsi(df)
    return df


def create_TechChart(df,ticker_name):
    # figを定義
    fig = make_subplots(rows=4, cols=1, shared_xaxes=False, row_heights=[6,2,2,2], x_title="Date")
    # Candlestick 
    fig.add_trace(
        go.Candlestick(x=df["datetime"], open=df["open"], high=df["high"], low=df["low"], close=df["close"], name="OHLC",
                ),
        row=1, col=1
    )
    # SMA
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["SMA5"], name="SMA5", mode="lines",
                             hovertemplate = "DATE:%{x}: <br>Prise($):%{y}",)
                  , row=1, col=1,
                  
                  )
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["SMA25"], name="SMA25", mode="lines",
                             hovertemplate = "DATE:%{x}: <br>Prise($):%{y}"),
                  row=1, col=1,
                  )
    
    # Volume
    fig.add_trace(
        go.Bar(x=df["datetime"], y=df["volume"], name="Volume",
               hovertemplate = "DATE:%{x}: <br>Volume:%{y}"
               ),
        row=2, col=1
    )
    
    # MACD
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["MACD"], name="MACD", mode="lines",
                             hovertemplate = "DATE:%{x}: <br>MACD:%{y}",
                  ), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["Signal"], name="Signal", mode="lines",
                             hovertemplate = "DATE:%{x}: <br>Signal:%{y}",
                  ), row=3, col=1,)
    
    # RSI
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["RSI"], name="RSI", mode="lines",
                             hovertemplate = "DATE:%{x}: <br>RSI:%{y}"
                             ), row=4, col=1)
    
    # Layout
    fig.update_layout(
        title={
            "text": "Technical Charts : {}".format(ticker_name),
            # "y":0.99,
            "x":0.5,
            # "size":40
        }
    )
    
    fig.update_layout(
                        # autosize=False,
                        # height=1000,
                    title=dict(
                        font=dict(size=25,family='Gravitas')
                        ),
                      hoverlabel=dict(font=dict(size=20)),
                      plot_bgcolor=fig_bgColor,
                      paper_bgcolor= web_bgColor,
                      # height=1800,
                      legend=dict(
                        traceorder="reversed",
                        title_font_family="Times New Roman",
                        font=dict(
                            family="Courier",
                            size=12,
                            color="black"
                            ),
                        bgcolor="ivory",
                        bordercolor="Black",
                        borderwidth=2
                        )
                      )
    
    # y軸名を定義
    fig.update_yaxes(title_text="株価", row=1, col=1)#,title_font_size=10,tickfont_size=10)
    fig.update_yaxes(title_text="出来高", row=2, col=1)#,title_font_size=10,tickfont_size=10)
    fig.update_yaxes(title_text="MACD", row=3, col=1)#,title_font_size=10,tickfont_size=10)
    fig.update_yaxes(title_text="RSI", row=4, col=1)#,title_font_size=10,tickfont_size=10)
    
    #日付リストを取得
    d_all = pd.date_range(start=df['datetime'].iloc[0],end=df['datetime'].iloc[-1])
    start=df['datetime'].iloc[0]
    end=df['datetime'].iloc[-1]
    #株価データの日付リストを取得
    d_obs = [d.strftime("%Y-%m-%d") for d in df['datetime']]
    
    # 株価データの日付データに含まれていない日付を抽出
    d_breaks = [d for d in d_all.strftime("%Y-%m-%d").tolist() if not d in d_obs]
    
    for i in [1,2,3,4]:
        fig.update_xaxes(range=[start,end],
            rangebreaks=[dict(values=d_breaks)],
            title_font_size=20,
            tickfont_size=15,
            row=i, col=1
        )
    fig.update(layout_xaxis_rangeslider_visible=False)
    return fig


def convert_militrillion(value_mlt):
    value_conved=[]
    for value in value_mlt:
        mlt = value[-1]
        if mlt =='M':
            value_conved.append(float(value[:-1])*(10**6))
        elif mlt == 'B':
            value_conved.append(float(value[:-1])*(10**9))
        else:
            value_conved.append(float(value[:-1]))
    return value_conved


def get_Est(ticker):
    df_all = pd.DataFrame()
    ticker_analysts_info = si.get_analysts_info(ticker)
    DATA_ReveEst = ticker_analysts_info["Revenue Estimate"]
    DATA_EarnEst = ticker_analysts_info["Earnings Estimate"]
    DATA_ReveEst = DATA_ReveEst.iloc[:,[0,1,3]]
    DATA_EarnEst = DATA_EarnEst.iloc[:,[0,1,3]]
    
    
    DATA_EarnEst.index=DATA_EarnEst['Earnings Estimate']
    DATA_EarnEst=DATA_EarnEst.drop('Earnings Estimate',axis=1)
    DATA_EarnEst= DATA_EarnEst.loc[['Avg. Estimate','Year Ago EPS']]
    if DATA_EarnEst.isnull().values.sum() == 0:
        # aaa=(DATA_EarnEst.loc['Avg. Estimate']/DATA_EarnEst.loc['Year Ago EPS'])
        DATA_EarnEst.columns=['Quarterly','Yearly']
        DATA_EarnEst.index=['Estimate','YearAgo']
        DATA_EarnEst_q = DATA_EarnEst['Quarterly']
        DATA_EarnEst_q.index = ['Estimate_EPS_q','YearAgo_EPS_q']
        DATA_EarnEst_y = DATA_EarnEst['Yearly']
        DATA_EarnEst_y.index = ['Estimate_EPS_y','YearAgo_EPS_y']
        df_all = pd.concat([df_all,DATA_EarnEst_q])
        df_all = pd.concat([df_all,DATA_EarnEst_y])
        l=[]
        for a,b in zip( DATA_EarnEst.loc['Estimate'], DATA_EarnEst.loc['YearAgo']):
            l.append(float(a)-float(b))
        l_rate=[]
        for a,b in zip( DATA_EarnEst.loc['Estimate'], DATA_EarnEst.loc['YearAgo']):
            l_rate.append((float(a)-float(b))/abs(float(b)))
        DATA_EarnEst=DATA_EarnEst.transpose()
        DATA_EarnEst['Growth_Estimate']= l
        DATA_EarnEst['GrowthRate_Estimate']= l_rate
        DATA_EarnEst.index=['年間EPS成長率見込み_Q','年間EPS成長率見込み_Y']
        df_all = pd.concat([df_all,DATA_EarnEst['GrowthRate_Estimate']])
        DATA_EarnEst.index=['年間EPS成長見込み_Q','年間EPS成長見込み_Y']
        df_all = pd.concat([df_all,DATA_EarnEst['Growth_Estimate']])
        
    
    DATA_ReveEst.index=DATA_ReveEst['Revenue Estimate']
    DATA_ReveEst=DATA_ReveEst.drop('Revenue Estimate',axis=1)
    DATA_ReveEst= DATA_ReveEst.loc[['Avg. Estimate','Year Ago Sales']]
    if DATA_ReveEst.isnull().values.sum() == 0:
        # DATA_ReveEst = DATA_ReveEst.transpose()
        DATA_ReveEst.columns=['Quarterly','Yearly']
        DATA_ReveEst.index=['Estimate','YearAgo']
        DATA_ReveEst['Quarterly'] = convert_militrillion(DATA_ReveEst['Quarterly'])
        DATA_ReveEst['Yearly'] = convert_militrillion(DATA_ReveEst['Yearly'])
        DATA_ReveEst_q = DATA_ReveEst['Quarterly']
        DATA_ReveEst_q.index = ['Estimate_Rev_q','YearAgo_Rev_q']
        DATA_ReveEst_y = DATA_ReveEst['Yearly']
        DATA_ReveEst_y.index = ['Estimate_Rev_y','YearAgo_Rev_y']
        df_all = pd.concat([df_all,DATA_ReveEst_q])
        df_all = pd.concat([df_all,DATA_ReveEst_y])
        l=[]
        for a,b in zip( DATA_ReveEst.loc['Estimate'], DATA_ReveEst.loc['YearAgo']):
            l.append(float(a)-float(b))
        l_rate=[]
        for a,b in zip( DATA_ReveEst.loc['Estimate'], DATA_ReveEst.loc['YearAgo']):
            l_rate.append((float(a)-float(b))/abs(float(b)))
        #[(float(a)-float(b))/abs(float(b)) for a,b in zip( DATA_ReveEst['Estimate'], DATA_ReveEst['YearAgo'])]
        DATA_ReveEst=DATA_ReveEst.transpose()
        DATA_ReveEst['Growth_Estimate'] = l
        DATA_ReveEst['GrowthRate_Estimate'] = l_rate
        DATA_ReveEst.index=['年間売上成長率見込み_Q','年間売上成長率見込み_Y']
        df_all = pd.concat([df_all,DATA_ReveEst['GrowthRate_Estimate']])
        DATA_ReveEst.index=['年間売上成長見込み_Q','年間売上成長見込み_Y']
        df_all = pd.concat([df_all,DATA_ReveEst['Growth_Estimate']])
    return df_all

df = get_Est('KO')
rev_data = df.loc[["年間売上成長見込み_Q",'年間売上成長見込み_Y'],:]
eps_data = df.loc[["年間EPS成長見込み_Q",'年間EPS成長見込み_Y'],:]
# rev_data = rev_data.round(2)
# rev_data = (list((rev_data.iloc[:,0]/10**6).round(-2)))
# DATA_EarnHist,DATA_Q = getStockInfo('DOCS')

#%% HTML layout

main_color = 'chocolate'
fig_bgColor='bisque'
web_bgColor='ivory'
value_bf=''
app = dash.Dash(__name__,suppress_callback_exceptions = True) 
app.layout = html.Div([
        html.H1('StockAnalizer KimChart',
                style={'textAlign': 'center',
                       'color':'snow',
                       # 'fontSize': 30,
                       'backgroundColor': main_color}
                
                ),
                
        html.Div([
            html.Div(
                children="Select target stocks..."
                ,style={'fontSize': 20}
                ),            
            dcc.Dropdown(
                id = 'stock_select',
                options=stock_dict_list,
                multi=True,
                value=['AAPL','AAL'],#,"AMZN",'FB','GOOG','MSFT'],
                # clearable=False
                style = {"background-color":fig_bgColor,
                          "border-color": 'red',
                          "color":main_color,
                          },
            )
        ]),
        
        html.Button('Get Stock Information', id='button',
                    style={
                        'fontSize': 20,
                        'background-color': main_color,
                        'color': web_bgColor,
                        'height': '30px',
                        'width': '300px',
                        'margin-top': '10px',
                        'margin-bottom': '20px',
                        'margin-left': '10px',
                    }),
        
        html.Div(id='push-button',
            children='Input ticker symbol and press submit',
            style={'fontSize': 20}
            ),
        
        html.Div(children="👇👇👇👇👇",
                style={'fontSize': 20,
                        # 'height': 'vh',
                        'backgroundColor': main_color}
        ),
        html.Div(id='output-container-button',
                 children='Enter a value and press submit',
                 style={'fontSize': 20,
                        'color':'snow',
                        'backgroundColor': main_color,
                        }),
        #左側
        html.Div([
            html.Div([
                
                        html.Div([
                            html.Div(
                                children="Select X-axis period..."
                                ,style={'fontSize': 20}
                                ),
                            dcc.Dropdown(
                                id = 'dwm_select',
                                options=[
                                    {'label': 'DAY', 'value': 'day'},
                                    {'label': 'WEEK', 'value': 'week'},
                                    {'label': 'MONTH', 'value': 'month'},
                                    {'label': 'YEAR', 'value': 'year'}
                                        ],
                                value='day',
                                clearable=False,
                                style = {"background-color":fig_bgColor,
                                         "color":main_color }
                            )
                        ]), 
                        
                        html.Div([
                            html.Div(
                                children="Select Y-axis period..."
                                ,style={'fontSize': 20}
                                ),
                            dcc.Dropdown(
                                id = 'dwm_select2',
                                options=[
                                    {'label': 'DAY', 'value': 'day'},
                                    {'label': 'WEEK', 'value': 'week'},
                                    {'label': 'MONTH', 'value': 'month'},
                                    {'label': 'YEAR', 'value': 'year'}
                                        ],
                                value='week',
                                clearable=False,
                                style = {"background-color":fig_bgColor,
                                         "color":main_color }
                            )
                        ]), 
        
                
                    html.Div([
                        dcc.Graph(id='scatter_chart',
                        style={
                            'width': '50vw',
                            'height': '70vh'
                            }),
        
                        ]),
                    
                        html.Div(id='ticker_name',
                             children='----',
                             style={'fontSize': 20,
                                    'backgroundColor': fig_bgColor,
                                    'height': '15vh'}),
                        
                        html.Div([
                            dcc.Graph(id='kessan_chart',
                            style={
                                'width': '50vw',
                                'height': '50vh'
                                }),
                            dcc.Graph(id='kessan_chart2',
                            style={
                                'width': '50vw',
                                'height': '50vh'
                                }
                            ),
                            dcc.Graph(id='kessan_chart3',
                            style={
                                'width': '50vw',
                                'height': '50vh'
                                }
                            )
            
                        ]),
                        
                ],style={'display': 'flex', 'flex-direction': 'column',
                     'height': '150vh','margin-top': '20px',}
                
                ),
                     
        #右側
        
        html.Div([
            dcc.RadioItems(
                id='period_select',
                options=[
                    {'label': 'Month', 'value': 1},
                    {'label': 'HalfYear', 'value': 6},
                    {'label': 'Year', 'value': 12},
                    {'label': '5Years', 'value': 60}
                ],
                value=1,
                persistence = True,
                style={ 'fontSize': 20, 'margin-left': '50px','margin-top': '20px'},
                labelStyle={'display': 'inline-block',"padding": "10px"}
            ),
            
            dcc.Graph(
                    id='chart_d',
                    style={
                    'height': '220vh',
                    'width': '50vw',
                    }
                ),
            ])
            
            
            
            ],style={'display': 'flex', 'flex-direction': 'row',
                     'margin-top': '10px',
                     'height': '200vh'
                 }
            )
    
    ],style = {'backgroundColor': web_bgColor}, # 背景色
    )

#%% CALLBACK                
@app.callback(
    dash.dependencies.Output('scatter_chart', 'figure'),
    [dash.dependencies.Input('dwm_select', 'value'),
     dash.dependencies.Input('dwm_select2', 'value'),
     [dash.dependencies.State('stock_select', 'value')],
     [dash.dependencies.Input('button', 'n_clicks')]]
)
def update_graph(dwm1,dwm2,slct_stock,ncli):
    plotdata = mk_plotData(slct_stock)
    if dwm2 == 'day' : rate = 500
    elif dwm2 == 'week' : rate = 250
    elif dwm2 == 'month' : rate = 100
    else:rate = 30
    
    fig = go.Figure()
    for n,s in zip(plotdata['ticker'],plotdata['up-down_'+dwm2]):
        x=plotdata[plotdata['ticker']==n]['up-down_'+dwm1]
        y=plotdata[plotdata['ticker']==n]['up-down_'+dwm2]
        fig.add_trace(go.Scatter(x = x,
                                 y = y,
                                 mode='markers+text',
                                 name=n,
                                 hovertemplate="増減：%{y}",
                                 marker={
                                    'size' : 30+rate*abs(s-1),
                                 }, 
                         )
              )
        
    fig.update_layout(
        title={
            "text": "Prise Up-Down for a specified period",
            "x":0.5,
            }
        )
        
    fig.update_layout(
                    title=dict(
                        font=dict(size=25,family='Old Standard TT')
                        ),
                      hoverlabel=dict(font=dict(size=20)),
                      paper_bgcolor= web_bgColor,
                      plot_bgcolor=fig_bgColor,
                      legend=dict(
                        traceorder="reversed",
                        title_font_family="Times New Roman",
                        font=dict(
                            family="Courier",
                            size=10,
                            color="black"
                            ),
                        bgcolor="ivory",
                        bordercolor="Black",
                        borderwidth=2
                        )
                      )
    fig.update_xaxes(title=dwm1+' up-down',title_font_family="Open Sans",title_font_size=20,tickfont_size=20)
    fig.update_yaxes(title=dwm2+' up-down',title_font_family="Open Sans",title_font_size=20,tickfont_size=20)
    # fig.show()
    return fig
           
@app.callback(
    dash.dependencies.Output('kessan_chart', 'figure'),
    [dash.dependencies.State('stock_select', 'value')],
     [Input('scatter_chart', 'hoverData')]
)
def update_graph(slct_stock,hoverData):
    n=(hoverData["points"][0]['curveNumber'])
    ticker = slct_stock[n]
    DATA_EarnHist,Q_Data = getStockInfo(ticker)
    DATA_EarnHist['time']=DATA_EarnHist.index
    DATA_EarnHist['EPS Actual']=DATA_EarnHist['EPS Actual'].astype(float)
    DATA_EarnHist['EPS Est.']=DATA_EarnHist['EPS Est.'].astype(float)
    fig = go.Figure()
    fig.add_trace(go.Bar(
            x=DATA_EarnHist['time'], y=DATA_EarnHist['EPS Actual'],
            name='Actual',
            marker_color='indianred', marker_line_color='maroon',
            marker_line_width=2.5, opacity=0.8,
            hovertemplate = "EPS:%{y}",
            text=DATA_EarnHist['EPS Actual'],
        ))
    # fig.add_trace(px.bar(DATA_EarnHist, x='time', y='EPS Actual',title="EPS",opacity=0.5)
    fig.add_trace(go.Bar(
            x=DATA_EarnHist['time'], y=DATA_EarnHist['EPS Est.'],
            name='Estimate',
            marker_color='lightsalmon', marker_line_color='tomato',
                  marker_line_width=2.5, opacity=0.8,
            hovertemplate = "EPS:%{y}",
            text=DATA_EarnHist['EPS Est.'],
        ))
    
    fig.update_layout(
        title={
            "text": "EPS",
            "x":0.5,
            }
        )
    
        
    fig.update_layout(
                    title=dict(
                        font=dict(size=25,family='Gravitas')
                        ),
                    plot_bgcolor=fig_bgColor,
                    paper_bgcolor= web_bgColor,
                    hoverlabel=dict(font=dict(size=20))
                    )
    fig.update_layout(barmode='group', xaxis_tickangle=-45,)
    fig.update_yaxes(title='Profit per stock')
    # fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
    #               marker_line_width=1.5, opacity=0.6)
    return fig
    
@app.callback(
    dash.dependencies.Output('kessan_chart2', 'figure'),
    [dash.dependencies.State('stock_select', 'value')],
     [Input('scatter_chart', 'hoverData')]
)
def update_graph(slct_stock,hoverData):
    n=(hoverData["points"][0]['curveNumber'])
    ticker = slct_stock[n]
    DATA_EarnHist,Q_Data = getStockInfo(ticker)
    Q_Data['time']=DATA_EarnHist.index
    Q_Data['総売上']=Q_Data['総売上'].astype(float)/10**6
    Q_Data['営業利益']=Q_Data['営業利益'].astype(float)/10**6
    Q_Data['営業利益率'] = Q_Data['営業利益']/Q_Data['総売上']*100
    max_ = Q_Data['総売上'].max()
    min_ = Q_Data['総売上'].min()
    delta = abs(max_ - min_)
    # fig = px.bar(Q_Data, x='time', y='総売上',title="総売上")
    fig = go.Figure()
    fig.add_trace(go.Bar(
            x=Q_Data['time'], y=Q_Data['総売上'],
            text=Q_Data['総売上'],
            name='総売上',
            yaxis='y1',
            # marker_color='gold',
            hovertemplate = "総売上:%{y}",
            marker_color='gold', marker_line_color='goldenrod',
            marker_line_width=2.5, opacity=0.6
        ))
    
    fig.add_trace(go.Bar(
            x=Q_Data['time'], y=Q_Data['営業利益'],
            text=Q_Data['営業利益'],
            name='営業利益',
            # marker_color='gold',
            hovertemplate = "営業利益:%{y}",
            yaxis='y1',
            marker_color='darkseagreen', marker_line_color='olivedrab',
            marker_line_width=2.5, opacity=0.6
        ))
    
    fig.add_trace(go.Line(
            x=Q_Data['time'], y=Q_Data['営業利益率'],
            text=Q_Data['営業利益率'],
            name='営業利益率',
            yaxis='y2',
            mode='lines+markers+text',
            marker=dict(color='orange',size=50,line_color='peru'),
            opacity=0.8,
            texttemplate='%{y:0.1f}%',
            # marker_color='gold',
            hovertemplate = "営業利益率:%{y}",
            textfont=dict(color='white')
        ))
    fig.update_traces(marker_line_width=3)
    
    fig.update_layout(
        title={
            "text": "Revenue & Profit",
            "x":0.5,
            },
        yaxis = dict(title = 'Revenue/Profit (M$)', side = 'left', showgrid=False),
        yaxis2 = dict(title = 'Return (%)',side = 'right',overlaying = 'y',  showgrid=False),
        )
    
    fig.update_layout(
                    title=dict(
                        font=dict(size=25,family='Gravitas')
                        ),
                    plot_bgcolor=fig_bgColor,
                    paper_bgcolor= web_bgColor,
                    hoverlabel=dict(font=dict(size=20))
                    )
    
    fig.update_layout(
                    # yaxis_range=[min_-delta*0.5,max_+delta*0.5],
                      xaxis_tickangle=-45,)
    return fig
    
@app.callback(
    dash.dependencies.Output('kessan_chart3', 'figure'),
    [dash.dependencies.State('stock_select', 'value')],
      [Input('scatter_chart', 'hoverData')]
)
def update_graph(slct_stock,hoverData):
    n=(hoverData["points"][0]['curveNumber'])
    ticker = slct_stock[n]
    est_data = get_Est(ticker)
    y_EPS_data = est_data.loc[["YearAgo_EPS_y",'Estimate_EPS_y',],:]
    y_Rev_data = est_data.loc[["YearAgo_Rev_y",'Estimate_Rev_y',],:]
    y_Rev_data = (y_Rev_data/10**6).astype(int)
    y_EPS_data = round(y_EPS_data,2)
    y_Rev_data = (list(y_Rev_data.iloc[:,0]))
    y_EPS_data = (list(y_EPS_data.iloc[:,0]))
    
    q_EPS_data = est_data.loc[["YearAgo_EPS_q",'Estimate_EPS_q'],:]
    q_Rev_data = est_data.loc[["YearAgo_Rev_q",'Estimate_Rev_q'],:]
    q_Rev_data = (q_Rev_data/10**6).astype(int)
    q_EPS_data = round(q_EPS_data,2)
    q_Rev_data = (list(q_Rev_data.iloc[:,0]))
    q_EPS_data = (list(q_EPS_data.iloc[:,0]))
    max_ = max(y_Rev_data)
    min_ = min(q_Rev_data)
    delta=abs(max_-min_)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
            x=['YearAgo','Estimate'], y=q_Rev_data,
            text=q_Rev_data,
            # textposition='auto',
            name='q_Revenue',
            yaxis='y1',
            hovertemplate = "q_Revenue:%{y}",
            # mode='lines+markers+text',
            # marker=dict(color='lightcyan',size=40),
            # opacity=0.8,
            marker_color='gold', marker_line_color='goldenrod',
            marker_line_width=2.5, opacity=0.6
        ))
    
    fig.add_trace(go.Bar(
            x=['YearAgo','Estimate'], y=y_Rev_data,
            text=y_Rev_data,
            # textposition='auto',
            name='y_Revenue',
            yaxis='y1',
            hovertemplate = "y_Revenue:%{y}",
            # mode='lines+markers+text',
            # marker=dict(color='darkcyan',size=40),
            # opacity=0.8,
            marker_color='paleturquoise', marker_line_color='darkcyan',
            marker_line_width=2.5, opacity=0.6
        ))
    
    fig.add_trace(go.Line(
            x=['YearAgo','Estimate'], y=y_EPS_data,
            text=y_EPS_data,
            # textposition='auto',
            name='y_EPS',
            yaxis='y2',
            hovertemplate = "y_EPS:%{y}",
            mode='lines+markers+text',
            marker=dict(color='violet',size=50, line_color='purple'),
            opacity=0.9,
            textfont=dict(color='white')
            # marker_color='pink', marker_line_color='violet',
            # marker_line_width=2.5, opacity=0.6
        ))
    fig.update_traces(marker_line_width=3)
    
    fig.add_trace(go.Line(
            x=['YearAgo','Estimate'], y=q_EPS_data,
            text=q_EPS_data,
            # textposition='auto',
            name='q_EPS',
            yaxis='y2',
            hovertemplate = "q_EPS:%{y}",
            mode='lines+markers+text',
            marker=dict(color='indianred',size=50, line_color='maroon'),
            opacity=0.9,
            textfont=dict(color='white')
            
            # marker_color='pink', marker_line_color='violet',
            # marker_line_width=2.5, opacity=0.6
        )),
    # fig.update_layout(
    #             font=dict(color='white')
    #             )
    fig.update_traces(marker_line_width=3),
    

    fig.update_layout(
        title={
            "text": "Growth Estimete of Revenue&EPS",
            "x":0.5,
            },
        yaxis = dict(title = 'Revenue (M$)', side = 'left', showgrid=False),
        yaxis2 = dict(title = 'EPS ($)',side = 'right',overlaying = 'y',  showgrid=False),
        ),

    fig.update_layout(
                    title=dict(
                        font=dict(size=25,family='Gravitas')
                        ),
                    plot_bgcolor=fig_bgColor,
                    paper_bgcolor= web_bgColor,
                    hoverlabel=dict(font=dict(size=20))
                    ),
    
    fig.update_layout(xaxis_tickangle=-45,)
    # fig.update_layout(yaxis=dict(range=[min_-delta, max_+delta]))
    # fig.update_layout(yaxis2=dict(range=[-10, 20]))
    return fig

@app.callback(
    Output("chart_d", 'figure'),
    [dash.dependencies.State('stock_select', 'value')],
    [dash.dependencies.Input('period_select', 'value')],
    [Input('scatter_chart', 'hoverData')]
)
def show_img(slct_stock,period_num,hoverData):
    n=(hoverData["points"][0]['curveNumber'])
    ticker = slct_stock[n]
    print(period_num)
    data = mk_Teckdata(ticker,period_num)
    return create_TechChart(data,ticker)
    

@app.callback(
    dash.dependencies.Output('output-container-button', 'children'),
    [dash.dependencies.Input('button', 'n_clicks')],
    [dash.dependencies.State('stock_select', 'value')])
def update_output(n_clicks, value):
    global value_bf
    if value_bf != value:
        value_bf = value
    return 'The inputed ticker is {}'.format(value)

@app.callback(
    dash.dependencies.Output('ticker_name', 'children'),
    [dash.dependencies.State('stock_select', 'value')],
    [Input('scatter_chart', 'hoverData')])
def update_output(slct_stock,hoverData):
    n=(hoverData["points"][0]['curveNumber'])
    ticker = slct_stock[n]
    ticker_business = str(stock_table[stock_table['symbol']==ticker].iloc[0,1])
    s = '決算情報 ： {} ({})👇👇👇'.format(ticker,ticker_business)
    return s


if __name__=="__main__":
    app.run_server(debug=False, use_reloader=False)