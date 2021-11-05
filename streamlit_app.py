import random

import alpaca_trade_api as tradeapi
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pandas as pd
import streamlit as st
import base64
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

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
    with _lock:
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

####
def get_bar_chart():
    with _lock:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        sorted_data = get_ticker_summary().todays_change.sort_values(ascending=False)
        ax.bar(sorted_data.index, sorted_data)
        plt.xticks([], [])
    return fig

####
def get_ticker_summary():
    if 'ticker_summary' not in st.session_state:
        filtered_symbols = list(pd.read_csv("https://raw.githubusercontent.com/gargshail/stocktrader/main/tickers.csv",
                                            index_col=0).index)

        # filtered_symbols = list(pd.read_csv("tickers.csv", index_col=0).index)
        wlist = get_watchlist().index
        for ticker in wlist:
            if ticker not in filtered_symbols:
                filtered_symbols.append(ticker)

        api = tradeapi.REST()
        batch_size = 100
        chunked = [filtered_symbols[i * batch_size:(i + 1) * batch_size] for i in
                   range((len(filtered_symbols) + batch_size - 1) // batch_size)]
        data = {}
        for chunk in chunked:
            bars = api.get_barset(chunk, "day", limit=250)
            for ticker in chunk:
                data[ticker] = bar_to_df(bars[ticker])

        for i in data:
            if data[i].close.count() < 210 or data[i].high.count() < 210:
                # print(f"{i} {data[i].close.count()}")
                try:
                    filtered_symbols.remove(i)
                except ValueError:
                    print(f"Already removed {i}")
                    continue

        for ticker in filtered_symbols:
            data[ticker]['ma50'] = data[ticker].close.rolling(50).mean()
            data[ticker]['vma50'] = data[ticker].volume.rolling(50).mean()
            data[ticker]['pchange'] = 100 * data[ticker].close.pct_change()
            data[ticker]['tight'] = data[ticker].pchange.abs() < 1
            data[ticker]['tightcount'] = data[ticker].groupby(
                (data[ticker]['tight'] != data[ticker]['tight'].shift(1)).cumsum()).cumcount() + 1
            data[ticker]['max10tightcount'] = data[ticker].tightcount.rolling(10).max()
            data[ticker]['volume_vma50_ratio'] = data[ticker].volume / data[ticker].vma50
            data[ticker]['volume_30pct_below'] = data[ticker]['volume_vma50_ratio'] < 0.7
            data[ticker]['low_volume_10day_count'] = data[ticker]['volume_30pct_below'].rolling(10).sum()
            data[ticker]['successive_low_volume_count'] = data[ticker].groupby((data[ticker]['volume_30pct_below'] != data[ticker]['volume_30pct_below'].shift(1)).cumsum()).cumcount() + 1
            data[ticker]['inside_day'] = (data[ticker].shift().high > data[ticker].high) & (data[ticker].shift().low < data[ticker].low)
        ticker_summary = pd.DataFrame(index=filtered_symbols)
        ticker_summary['open'] = pd.Series({t: data[t].open[-1] for t in filtered_symbols})
        ticker_summary['high'] = pd.Series({ticker: data[ticker].high[-1] for ticker in filtered_symbols})
        ticker_summary['low'] = pd.Series({ticker: data[ticker].low[-1] for ticker in filtered_symbols})
        ticker_summary['close'] = pd.Series({ticker: data[ticker].close[-1] for ticker in filtered_symbols})
        ticker_summary['volume'] = pd.Series({ticker: data[ticker].volume[-1] for ticker in filtered_symbols})
        ticker_summary['ma50'] = pd.Series({ticker: data[ticker].ma50[-1] for ticker in filtered_symbols})
        ticker_summary['vma50'] = pd.Series({ticker:int(data[ticker].vma50[-1]) for ticker in filtered_symbols})
        ticker_summary['todays_change'] = pd.Series(
            {ticker: data[ticker].tail(2).close.pct_change().iloc[1] * 100 for ticker in filtered_symbols})
        ticker_summary['maxmin_10day_perct'] = pd.Series({ticker: 100 * (data[ticker].close.tail(10).max()
                                                                         - data[ticker].close.tail(10).min()) / data[
                                                                      ticker].close.tail(10).min()
                                                          for ticker in filtered_symbols})
        ticker_summary['maxmin_5day_perct'] = pd.Series({ticker: 100 * (data[ticker].close.tail(5).max()
                                                                        - data[ticker].close.tail(5).min()) / data[
                                                                     ticker].close.tail(5).min()
                                                         for ticker in filtered_symbols})
        ticker_summary['maxmin_3day_perct'] = pd.Series({ticker: 100 * (data[ticker].close.tail(3).max()
                                                                        - data[ticker].close.tail(3).min()) / data[
                                                                     ticker].close.tail(3).min()
                                                         for ticker in filtered_symbols})
        ticker_summary['max10tightcount'] = pd.Series(
            {ticker: data[ticker].max10tightcount[-1] for ticker in filtered_symbols})
        ticker_summary['low_volume_10day_count'] = pd.Series(
            {ticker: data[ticker]['low_volume_10day_count'][-1] for ticker in filtered_symbols})
        ticker_summary['successive_low_volume_count'] = pd.Series(
            {ticker: data[ticker]['successive_low_volume_count'][-1] for ticker in filtered_symbols})
        ticker_summary['inside_day'] = pd.Series(
            {ticker: data[ticker]['inside_day'][-1] for ticker in filtered_symbols})
        st.session_state['ticker_summary'] = ticker_summary
    return st.session_state['ticker_summary']

def get_watchlist():
    url = 'https://raw.githubusercontent.com/gargshail/stocktrader/main/watchlist.csv'
    # url = "watchlist.csv"
    watchlist = pd.read_csv(url, index_col=0)
    # api = tradeapi.REST()
    # bars = api.get_barset(watchlist.index, "day", limit=2).df
    # for wticker in list({a[0] for a in bars.columns.values}):
    #     bars[wticker, 'todays_change'] = bars[wticker].close.pct_change()
    # df = pd.DataFrame(index=list({a[0] for a in bars.columns.values}))
    # df['todays_change'] = [bars[ticker, 'todays_change'].tail(1).values[0] for ticker in df.index]
    # watchlist = watchlist.join(df)
    return watchlist

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=True)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


def get_alerts():
    wlist = get_watchlist()
    ticker_summary = get_ticker_summary()
    combined = wlist.join(ticker_summary)
    # st.dataframe(combined)
    triggered_index = []
    for i, row in combined.iterrows():
        cols = row.index
        trigger = row['trigger']
        for c in cols:
            trigger = trigger.replace(c, f"row.{c}")
        if eval(trigger):
            triggered_index.append(i)
    return combined.loc[triggered_index]


def get_vcp_list(max10tightcount=4, low_volume_10day_count=4, maxmin_10day_perct=5, successive_low_volume_count=1, inside_day=False):
    ticker_summary = get_ticker_summary()
    vcp_stocks = ticker_summary[
        (ticker_summary['maxmin_10day_perct'] <= maxmin_10day_perct)
        & (ticker_summary['max10tightcount'] >= max10tightcount)
        & (ticker_summary['low_volume_10day_count'] >= low_volume_10day_count)
        & (ticker_summary['successive_low_volume_count'] >= successive_low_volume_count)
        ]
    if inside_day:
        return vcp_stocks[vcp_stocks['inside_day']]
    return vcp_stocks

display_charts = st.sidebar.checkbox('Display Charts', True)
tight_days = st.sidebar.slider("tightDays", 0, 10,4)
low_volume_days = st.sidebar.slider("lowVolumeDays", 0, 10,4)
slvc = st.sidebar.slider("successive low volume count", 0, 10, 1)
inside_day = st.sidebar.checkbox("InsideDay")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(get_chart('SPY'))

with col2:
    st.write(get_chart('DIA'))

with col3:
    st.write(get_chart('QQQ'))


sp500 = get_sp500_df()

st.markdown("# VCP List")
vcp_list = get_vcp_list(tight_days, low_volume_days, successive_low_volume_count=slvc, inside_day=inside_day)
# watchlist = get_watchlist()

if display_charts:
    tickers_for_chart = vcp_list.index
    images = []
    for ticker in tickers_for_chart:
        images.append(f"https://finviz.com/chart.ashx?t={ticker}&ta=1&p=d&s=m&rev={random.random() * 1000}")
    st.image(images)

st.dataframe(vcp_list)
st.write(f"{list(vcp_list.index)}".replace("'", "").replace("[","").replace("]",""))
st.markdown(get_table_download_link(vcp_list), unsafe_allow_html=True)

st.markdown("## Alerts")
alerts = get_alerts()
st.dataframe(alerts)
tickers_in_alert = alerts.index
alert_charts = []
for ticker in tickers_in_alert:
    alert_charts.append(f"https://finviz.com/chart.ashx?t={ticker}&ta=1&p=d&s=m&rev={random.random() * 1000}")
st.image(alert_charts)
