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
import matplotlib.pyplot as plt


# all_stocks = api.list_assets(status='active')
# etf_list = [ticker.symbol for ticker in all_stocks if (ticker.name.lower().find(" etf") != -1 
#                                                        or ticker.name.lower().find('trust') != -1 
#                                                        or ticker.name.lower().find('fund') != -1 
#                                                       ) ]

def bar_to_df(bars) :
    df = pd.DataFrame({
        'open':[bar.o for bar in bars],
        'high':[bar.h for bar in bars],
        'low':[bar.l for bar in bars],
        'close':[bar.c for bar in bars],
        'volume':[bar.v for bar in bars]
    }, index=[bar.t for bar in bars])
    return df

api = tradeapi.REST()
st.write("Test 1 ")
url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
sp500 = pd.read_csv(url, index_col=0)
st.dataframe(sp500)

tickers_for_chart = ["INTC", "SPY"]
for ticker in tickers_for_chart:
    st.image(f"https://finviz.com/chart.ashx?t={ticker}&ta=1&p=d&s=m&rev={random.random()*1000}") 
    

tickers = sp500.index
batch_size = 100
chunked = [tickers[i * batch_size:(i + 1) * batch_size] for i in range((len(tickers) + batch_size - 1) // batch_size )]
data = {}
for chunk in chunked:
    bars = api.get_barset(chunk, "day", limit=2)
    for ticker in chunk:
        data[ticker] = bar_to_df(bars[ticker])
ticker_summary = pd.DataFrame(index=tickers)
ticker_summary['todays_change'] = pd.Series({ticker: data[ticker].tail(2).close.pct_change().iloc[1]*100 for ticker in tickers})

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sorted_data = ticker_summary.todays_change[tickers].sort_values(ascending=False)
ax.bar(sorted_data.index, sorted_data)
st.write(fig)

st.dataframe(ticker_summary)

up_count = ticker_summary.query('todays_change > 0').count()[0]
down_count = ticker_summary.query('todays_change < 0').count()[0]

st.markdown(f"Up:{up_count}")
st.markdown(f"Down:{down_count}")

