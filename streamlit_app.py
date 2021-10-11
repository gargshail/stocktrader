import random

import alpaca_trade_api as tradeapi
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import streamlit as st
st.set_page_config(
    page_title="Stocktrader",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )
plt.style.use('seaborn-whitegrid')
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
np.set_printoptions(precision=4,suppress=True)

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

@st.cache
def get_sp500_df():
    url = 'https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv'
    return pd.read_csv(url, index_col=0)

def get_chart(ticker):
    api = tradeapi.REST()
    bars = api.get_barset([ticker], "5Min", limit=90).df
    recent_market_date = str(numpy.datetime_as_string(bars[ticker].tail(1).index.values[0], unit='D'))
    daybefore_market_date = str(numpy.datetime_as_string(bars[ticker].head(1).index.values[0], unit='D'))
    yesterday_close = bars[ticker].loc[daybefore_market_date].close.tail(1).values[0]
    market_data_df = pd.DataFrame()
    market_data_df[ticker] = 100*(bars[ticker].close - yesterday_close)/yesterday_close
    last_val = market_data_df[ticker].loc[recent_market_date].iloc[-1]
    fig = plt.figure(figsize=[8,8])
    ax1 = fig.add_subplot(211)
    xvals = market_data_df.loc[recent_market_date].index
    yvals = market_data_df.loc[recent_market_date]
    ax1.plot(xvals, yvals, "-" )
    ax1.axhline(0, linestyle="--")
    ax1.set_title(ticker, loc='left', fontweight='bold')
    ax1.set_title(recent_market_date, loc='center', fontsize=9)
    value_color="green" if last_val >= 0 else "red"
    ax1.set_title(f"{round(last_val,2)}%", loc='right', color=value_color)
    ax1.grid(color='grey', linestyle='--', alpha=0.3)
    return fig

def get_sp500_bar_chart():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    sorted_data = ticker_summary.todays_change.sort_values(ascending=False)
    ax.bar(sorted_data.index, sorted_data)
    return fig

@st.cache
def get_ticker_summary():
    api = tradeapi.REST()
    tickers = get_sp500_df().index
    batch_size = 100
    chunked = [tickers[i * batch_size:(i + 1) * batch_size] for i in
               range((len(tickers) + batch_size - 1) // batch_size)]
    data = {}
    for chunk in chunked:
        bars = api.get_barset(chunk, "day", limit=2)
        for ticker in chunk:
            data[ticker] = bar_to_df(bars[ticker])
    ticker_summary = pd.DataFrame(index=tickers)
    ticker_summary['todays_change'] = pd.Series(
        {ticker: data[ticker].tail(2).close.pct_change().iloc[1] * 100 for ticker in tickers})
    return ticker_summary

col1, col2, col3 = st.columns(3)
with col1:
    st.write(get_chart('SPY'))

with col2:
    st.write(get_chart('DIA'))

with col3:
    st.write(get_chart('QQQ'))


sp500 = get_sp500_df()
tickers_for_chart = ["INTC", "SPY"]
for ticker in tickers_for_chart:
    st.image(f"https://finviz.com/chart.ashx?t={ticker}&ta=1&p=d&s=m&rev={random.random()*1000}") 
    
ticker_summary = get_ticker_summary()
st.dataframe(sp500.join(ticker_summary))
up_count = ticker_summary.query('todays_change > 0').count()[0]
down_count = ticker_summary.query('todays_change < 0').count()[0]

st.markdown(f"Up:{up_count}")
st.markdown(f"Down:{down_count}")

st.write(get_sp500_bar_chart())
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# sorted_data = ticker_summary.todays_change.sort_values(ascending=False)
# ax.bar(sorted_data.index, sorted_data)
# # ax.set_xticklabels([])
# st.write(fig)

