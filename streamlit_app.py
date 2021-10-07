from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import alpaca_trade_api as tradeapi
import pandas as pd
import random
import datetime
import numpy


# all_stocks = api.list_assets(status='active')
# etf_list = [ticker.symbol for ticker in all_stocks if (ticker.name.lower().find(" etf") != -1 
#                                                        or ticker.name.lower().find('trust') != -1 
#                                                        or ticker.name.lower().find('fund') != -1 
#                                                       ) ]

st.write("Test 1 ")
url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
sp500 = pd.read_csv(url, index_col=0)
st.dataframe(sp500)

tickers_for_chart = ["INTC", "SPY"]
for ticker in tickers_for_chart:
    st.image(f"https://finviz.com/chart.ashx?t={ticker}&ta=1&p=d&s=m&rev={random.random()*1000}") 
    


