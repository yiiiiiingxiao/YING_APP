# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:34:43 2023

@author: yxiao
"""

# -*- coding: utf-8 -*-
###############################################################################
# FINANCIAL DASHBOARD
###############################################################################
# cd C:\Users\yxiao\Desktop\Python - Financial Programming\Individual Assignment
# python -m streamlit run Ying.py
#==============================================================================
# Initiating
#==============================================================================

# Libraries
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import streamlit as st

#==============================================================================
# HOT FIX FOR YFINANCE .INFO METHOD
# Ref: https://github.com/ranaroussi/yfinance/issues/1729
#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "indexTrend,"
                         "defaultKeyStatistics"
                         )

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret
#==============================================================================
# Header
#==============================================================================
 # Header (or Sidebar):
 # - A dropdown menu to select the stock names from S&P 500.
 # - An “Update” button to download and update stock data.
 
def render_header():
    """
    This function renders the header of the dashboard with the following items:
        - Dropdown menu to select stock names from S&P 500.
        - Update button to download and update stock data.
    """
    
    # Add dashboard title and description
    st.title("💰FINANCIAL DASHBOARD📈 ")
    col1, col2 = st.columns([1,5])
    col1.write("Data source:")
    col2.image('./img/yahoo_finance.png', width=100)
    
    
    # Add the dropdown menu for selecting stock names from S&P 500
    global selected_ticker  # Set this variable as global, so the functions in all of the tabs can read it
    ticker = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']
    selected_ticker = st.selectbox("Select Ticker", ticker)
   
   
    # Add the "Update" button
    global update_button  # Set this variable as global, so the functions in all of the tabs can read it
    update_button = st.button("Update Data")
    
def update_data():
    # Perform actions when the "Update" button is clicked
    if update_button:
        return YFinance(selected_ticker)
#==============================================================================
# Tab 1
#==============================================================================

# Inside render_tab1() function
def render_tab1():
    """
    This function renders Tab 1 - Summary of the dashboard.
    """
    # Show to stock image
    col1, col2, col3 = st.columns([1, 3, 1])
    col2.image('./img/stock_market.jpg', use_column_width=True,
               caption='Company Stock Information')
    
    # Get the company information
    @st.cache_data
    def GetCompanyInfo(selected_ticker):
        """
        This function get the company information from Yahoo Finance.
        """        
        return YFinance(selected_ticker).info
        #return yf.Ticker(ticker).info
    
    # If the ticker is already selected
    if selected_ticker != '':
        # Get the company information in list format
        info = GetCompanyInfo(selected_ticker)
       
        # Show the company description using markdown + HTML
        st.subheader('**1. Business Summary:**')
        st.markdown('<div style="text-align: justify;">' + \
                    info['longBusinessSummary'] + \
                    '</div><br>',
                    unsafe_allow_html=True)
        
        st.subheader('**2. Key Info:**')
        col1, col2 = st.columns(2)
        # statistic
        statistic_keys = {'previousClose':'Previous Close',
                     'open'         :'Open',
                     'bid'          :'Bid',
                     'ask'          :'Ask',
                     'marketCap'    :'Market Cap',
                     'volume'       :'Volume'}
        company_stats = {}  # Dictionary
        for key in statistic_keys:
            company_stats.update({statistic_keys[key]:info[key]})
        company_stats = pd.DataFrame({'Value':pd.Series(company_stats)})  # Convert to DataFrame
        col1.dataframe(company_stats)
        
        
        # Fetch major holders, institutional holders, and mutual fund holders
        major_holders = yf.Ticker(selected_ticker).major_holders
        institutional_holders = yf.Ticker(selected_ticker).institutional_holders
        mutual_fund_holders = yf.Ticker(selected_ticker).mutualfund_holders
        
        # Display shareholders information
        st.write("**Major Holders**")
        st.dataframe(major_holders)
    
        st.write("**Institutional Holders**")
        st.dataframe(institutional_holders)
    
        st.write("**Mutual Fund Holders**")
        st.dataframe(mutual_fund_holders)
         
        # Show the company stock price
        st.subheader('**3. Stock Price Chart:**')
        # Add a dropdown menu to select the time duration
        time_duration = st.selectbox("Select Time Duration:", ["1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y", "MAX"])
        
        # Fetch and display stock information based on the selected duration
    @st.cache_data
    def GetStockSummary(selected_ticker, time_duration):
        selected_stock = yf.Ticker(selected_ticker)
        
           
        if time_duration == "1M":
            stock_info = selected_stock.history(period="1mo", interval="1d")
        elif time_duration == "3M":
            stock_info = selected_stock.history(period="3mo", interval="1d")
        elif time_duration == "6M":
            stock_info = selected_stock.history(period="6mo", interval="1d")
        elif time_duration == "YTD":
            stock_info = selected_stock.history(period="ytd", interval="1d")
        elif time_duration == "1Y":
            stock_info = selected_stock.history(period="1y", interval="1d")
        elif time_duration == "3Y":
            stock_info = selected_stock.history(period="3y", interval="1d")
        elif time_duration == "5Y":
            stock_info = selected_stock.history(period="5y", interval="1d")
        elif time_duration == "MAX":
            stock_info = selected_stock.history(period="max", interval="1d")
        
        # Display stock information
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=stock_info.index, y=stock_info['Close'], mode='lines', name='Close Price'))
        fig.update_layout(title_text="Stock Price Chart ("+time_duration+")", xaxis_title='Date', yaxis_title='Stock Price (USD)')
        st.plotly_chart(fig, use_container_width=True)
        
    # Call the function
    GetStockSummary(selected_ticker, time_duration)


#==============================================================================
# Tab 2
#==============================================================================

def render_tab2():
    """
    This function renders Tab 2 - Chart of the dashboard.
    """
    # Begin and end dates
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    end_date = col2.date_input("End date", datetime.today().date())
    
    # Add a dropdown menu to select the time duration, time intervals, plot type
    col1,col2 = st.columns(2)
    time_interval = col1.radio("Select Time Interval", ["1d", "1wk", "1mo"])
    plot_type = col2.radio("Select Plot Type", ["Line Plot", "Candle Plot"])
    
    # Fetch stock price data based on selected options
    @st.cache_data
    def GetStockData(selected_ticker,  time_interval, start_date, end_date):
        stock_df = yf.Ticker(selected_ticker).history(interval=time_interval, start=start_date, end=end_date)
        return stock_df
    
    # Add checkbox for Trading volumne and moving average
    col1, col2, col3 = st.columns(3)
    show_volume = col1.checkbox("Show Trading Volume")
    show_ma = col2.checkbox("Show Moving Average (MA)")
    # Add a check box to show/hide data
    show_data = col3.checkbox("Show Data Table")
    if selected_ticker != '':
        stock_price = GetStockData(selected_ticker, time_interval, start_date, end_date)
        if show_data:
            st.write('**Stock Price Data**')
            st.dataframe(stock_price, hide_index=True, use_container_width=True)
        
    # Plot the stock chart based on selected options      
    fig = go.Figure()
    
    if plot_type == "Line Plot":
        fig.add_trace(go.Scatter(x=stock_price.index, y=stock_price['Close'], mode='lines', name='Close Price'))
    elif plot_type == "Candle Plot":
        fig.add_trace(go.Candlestick(x=stock_price.index,
                                     open=stock_price['Open'],
                                     high=stock_price['High'],
                                     low=stock_price['Low'],
                                     close=stock_price['Close'],
                                     name='Stock Price'))

    if show_volume:
        fig.add_trace(go.Bar(x=stock_price.index, y=stock_price['Volume'], name='Volume'))

    if show_ma:
        ma_window = 50
        stock_price['MA'] = stock_price['Close'].rolling(window=ma_window).mean()
        fig.add_trace(go.Scatter(x=stock_price.index, y=stock_price['MA'], mode='lines', name=f'MA {ma_window} days'))

    # Customize layout
    fig.update_layout(title_text="Stock Price Chart", xaxis_title='Date', yaxis_title='Stock Price (USD)')

    # Display the chart
    st.plotly_chart(fig, use_container_width=True)


#==============================================================================
# Tab 3
#==============================================================================

def render_tab3():
    """
    This function renders Tab 3 - Financials of the dashboard.
    """
    
    # Add options for financial statements and period
    col1, col2 = st.columns(2)
    statement_type = col1.radio("Select Financial Statement", ["Income Statement", "Balance Sheet", "Cash Flow"])
    period_type = col2.radio("Select Period", ["Annual", "Quarterly"])

    # Fetch financial data based on selected options
    @st.cache_data
    def GetFinancialData(selected_ticker, statement_type, period_type):
        selected_stock = yf.Ticker(selected_ticker)
        financial_data = None

        if statement_type == "Income Statement":
            if period_type == "Annual":
                financial_data = selected_stock.financials
            elif period_type == "Quarterly":
                financial_data = selected_stock.quarterly_financials
        elif statement_type == "Balance Sheet":
            if period_type == "Annual":
                financial_data = selected_stock.balance_sheet
            elif period_type == "Quarterly":
                financial_data = selected_stock.quarterly_balance_sheet
        elif statement_type == "Cash Flow":
            if period_type == "Annual":
                financial_data = selected_stock.cashflow
            elif period_type == "Quarterly":
                financial_data = selected_stock.quarterly_cashflow

        return financial_data
    
    # Fetch and display financial data
    financial_data = GetFinancialData(selected_ticker, statement_type, period_type)
    if financial_data is not None:
        st.write(f'**{period_type} {statement_type}**')
        st.dataframe(financial_data.transpose(), use_container_width=True)


#==============================================================================
# Tab 4
#==============================================================================

def render_tab4():
    """
    This function renders Tab 4 - Monte Carlo of the dashboard.
    """

    # Add dropdowns for number of simulations and time horizon
    col1, col2 = st.columns(2)
    num_simulations = col1.selectbox("Number of Simulations", [200, 500, 1000])
    time_horizon = col2.selectbox("Time Horizon (days)", [30, 60, 90])

    # Fetch historical data for Monte Carlo simulation
    historical_data = yf.Ticker(selected_ticker).history(period="max", interval="1d")
    timeHorizonValues=[]

    # Perform Monte Carlo simulation
    @st.cache_data
    def MonteCarloSimulation(data, num_simulations, time_horizon):
        returns = data['Close'].pct_change().dropna()
        close_price = data['Close'].iloc[-1]

        simulation_df = pd.DataFrame()

        for i in range(num_simulations):
            daily_vol = returns.std()
            daily_drift = returns.mean()

            price_series = []

            last_price=close_price
            
            for day in range(1, time_horizon):
                last_price *= (1+np.random.normal(loc=daily_drift, scale=daily_vol))
                price_series.append(last_price)

            timeHorizonValues.append(last_price)
            
            simulation_df[f"Simulation {i+1}"] = price_series
            

        return simulation_df

    # Run Monte Carlo simulation
    simulation_results = MonteCarloSimulation(historical_data, num_simulations, time_horizon)

    # Calculate VaR at 95% confidence interval
    var_95 = np.percentile(timeHorizonValues, 5)

    # Plot simulation results
    st.write('**Monte Carlo Simulation Results**')
    fig = go.Figure()

    for i in range(num_simulations):
        fig.add_trace(go.Scatter(x=simulation_results.index, y=simulation_results[f"Simulation {i+1}"], mode='lines'))

    # Highlight the 95% confidence interval
    fig.add_shape(type='line',
                  x0=-1,
                  x1=simulation_results.index[-1]+1,
                  y0=var_95,
                  y1=var_95,
                  line=dict(color='red', dash='dash'),
                  name='95% Confidence Interval')
    
    fig.add_shape(type='line',
                  x0=-1,
                  x1=simulation_results.index[-1]+1,
                  y0=historical_data['Close'].iloc[-1],
                  y1=historical_data['Close'].iloc[-1],
                  line=dict(color='red'),
                  name='Closing Price')

    fig.update_layout(title_text=f"Monte Carlo Simulation ({num_simulations} simulations, {time_horizon} days)",
                      xaxis_title='Days',
                      yaxis_title='Stock Price (USD)',     
                      )

    st.plotly_chart(fig, use_container_width=True)

    # Display VaR at 95% confidence interval
    st.write(f'**Value at Risk (VaR) at 95% Confidence Interval:** ${var_95:.2f}')

#==============================================================================
# Tab 5
#==============================================================================
def render_tab5():
    """
    This function renders Tab 5 - Options and News of the dashboard.
    """
    
    # Fetch news
    stock_news = yf.Ticker(selected_ticker).news

    # Display stock news
    st.write("**Latest News**")
    st.dataframe(stock_news)

    # Fetch options expirations
    options_expirations = yf.Ticker(selected_ticker).options

    # Add an option to select a specific expiration date for the option chain
    selected_expiration = st.selectbox("Select Options Expiration", options_expirations)

    # Fetch option chain for the selected expiration
    option_chain = yf.Ticker(selected_ticker).option_chain(selected_expiration)

    # Display option chain data (calls and puts)
    st.write(f"**Option Chain Analysis for {selected_expiration}**")

    # Analyze and display statistics for calls
    st.subheader("**Calls Analysis**")
    st.write(f"Total Calls: {len(option_chain.calls)}")
    st.write(f"Average Call Price: ${option_chain.calls['lastPrice'].mean():.2f}")
    st.write(f"Max Call Price: ${option_chain.calls['lastPrice'].max():.2f}")
    st.write(f"Min Call Price: ${option_chain.calls['lastPrice'].min():.2f}")

    # Analyze and display statistics for puts
    st.subheader("**Puts Analysis**")
    st.write(f"Total Puts: {len(option_chain.puts)}")
    st.write(f"Average Put Price: ${option_chain.puts['lastPrice'].mean():.2f}")
    st.write(f"Max Put Price: ${option_chain.puts['lastPrice'].max():.2f}")
    st.write(f"Min Put Price: ${option_chain.puts['lastPrice'].min():.2f}")

    # Add more analysis based on your requirements

    # Display raw data if desired
    show_raw_data = st.checkbox("Show Raw Data")
    if show_raw_data:
        st.write("**Raw Option Chain Data (Calls)**")
        st.dataframe(option_chain.calls)

        st.write("**Raw Option Chain Data (Puts)**")
        st.dataframe(option_chain.puts)


#==============================================================================
# Main body
#==============================================================================
      
# Render the header
render_header()

# Render the tabs
tab1, tab2,tab3, tab4, tab5 = st.tabs(["Company profile", "Chart", "Financials","Monte Carlo", "Options and News"])
with tab1:
    render_tab1()
with tab2:
    render_tab2()
with tab3:
    render_tab3()
with tab4:
    render_tab4()
with tab5:
    render_tab5()

    
# Customize the dashboard with CSS
st.markdown(
    """
    <style>
        .stApp {
            background: #F0F8FF;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
    
###############################################################################
# END
###############################################################################
