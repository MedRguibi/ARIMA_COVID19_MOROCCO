import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

import seaborn as sns
import lightgbm as lgb ########### pip install lightgbm 
from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error

from scipy import stats as sps
from scipy.interpolate import interp1d

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from IPython.display import clear_output
from datetime import timedelta
from time import time

import warnings
warnings.filterwarnings("ignore")


font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

mpl.rc('font', **font)

mpl.rcParams['axes.grid']=True
plt.rcParams.update({'figure.figsize':(8, 5), 'figure.dpi':120})

mpl.rcParams['axes.grid']=True
pd.options.display.max_rows = 999

########################################## Import Actual Data ##################################

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')
recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')


def prepare(df, name):
    cols_to_melt = df.columns.values[4:]
    res = pd.melt(df, id_vars='Country/Region', var_name='date', value_vars=cols_to_melt, value_name=name)
    res['date'] = pd.to_datetime(res['date'])
    res = res.sort_values(by = ['Country/Region', 'date'])
    res = res.set_index('date')
    res.columns = ['country', name]
    return res

confirmed_df = prepare(confirmed_df, 'Confirmed Cases')
recovered_df = prepare(recovered_df, 'Recovered Cases')
death_df = prepare(death_df, 'Death Cases')

######################################################################################################################
################################# Auto Regressor Intergrated Moving average ##########################################
#Define function

def get_country(df, country):
    country_df = df[(df['country'] == country)]
    #country_df = country_df.set_index(keys = 'Date')
    return country_df

def rmsle(y, y_hat):
    y_hat = y_hat.clip(0)
    res = np.sqrt(mean_squared_log_error(y, y_hat))
    return res

def mean_absolute_percentage_error(actual, prediction):
    return 100 * np.mean(np.abs((actual - prediction))/actual)

def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.8)
    train, val = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(val)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(val[t])
    error = rmsle(val, np.array(predictions))
    return error

def evaluate_metrics(X, arima_order):
    train_size = int(len(X) * 0.8)
    train, val = X[0:train_size], X[train_size:]
    history = [x for x in train]
    predictions = list()
    for t in range(len(val)):
        model = ARIMA(history, order = arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(val[t])
    error_mse = mean_squared_error(val, np.array(predictions))
    error_rmse = error_mse ** 0.5
    error_msle = mean_squared_log_error(val, np.array(predictions))
    error_mae = mean_absolute_error(val, np.array(predictions))
    error_mape = mean_absolute_percentage_error(val, np.array(predictions))
    return error_mse, error_rmse, error_msle, error_mae, error_mape

def get_arima_order(dataset, p_values, d_values, q_values, target):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmsle = evaluate_arima_model(dataset, order)
                    if rmsle < best_score:
                        best_score, best_cfg = rmsle, order
                except:
                    continue
    print('>>>', target, 'Best ARIMA%s RMSLE = %.6f' % (best_cfg, best_score))
    mse ,rmse ,msle, mae, mape = evaluate_metrics(dataset,best_cfg)
    print('>>>', target, 'MSE = %.3f RMSE = %.3f MSLE = %.3f MAE = %.3f MAE = %.3f' % (mse, rmse, msle, mae, mape))
    return best_cfg

p_values = [0, 1, 2, 3]
d_values = range(0, 3)
q_values = range(0, 3)

def model_per_country(country, show_plots = False, days_num = 90):
    
    country_df_ConfirmedCases = get_country(confirmed_df, country)
    # country_df_RecoveredCases = get_country(recovered_df, country)
    country_df_Death = get_country(death_df, country)   

    arima_order_cc = get_arima_order(country_df_ConfirmedCases['Confirmed Cases'].values, p_values, d_values, q_values, 'Confirmed Cases')
    model_cc = ARIMA(country_df_ConfirmedCases['Confirmed Cases'], order = arima_order_cc, freq = 'D')
    fitted_model_cc = model_cc.fit(disp = 0)
    residuals_cc = pd.DataFrame(fitted_model_cc.resid)
    res_ConfirmedCases = fitted_model_cc.predict(arima_order_cc[1], len(country_df_ConfirmedCases) + days_num)

    # arima_order_rv = get_arima_order(country_df_RecoveredCases['Recovered Cases'].values, p_values, d_values, q_values, 'Recovered Cases')
    # model_rv = ARIMA(country_df_RecoveredCases['Recovered Cases'], order = arima_order_rv, freq = 'D')
    # fitted_model_rv = model_rv.fit(disp = 0)
    # res_RecoveredCases = fitted_model_rv.predict(arima_order_rv[1], len(country_df_RecoveredCases) + days_num)

    arima_order_fa = get_arima_order(country_df_Death['Death Cases'].values, p_values, d_values, q_values, 'Death Cases')
    model_fa = ARIMA(country_df_Death['Death Cases'], order = arima_order_fa, freq = 'D')
    fitted_model_fa = model_fa.fit(disp = 0)
    residuals_fa = pd.DataFrame(fitted_model_fa.resid)
    res_DeathCases = fitted_model_fa.predict(arima_order_fa[1], len(country_df_Death) + days_num)
   
    if show_plots:
        fitted_model_cc.plot_predict(arima_order_cc[1], len(country_df_ConfirmedCases) + days_num)
        residuals_cc.plot(title="Confirmed Cases Residuals", legend=None)
        residuals_cc.plot(kind='kde', title='Confirmed Cases Density', legend=None)
        plot_acf(residuals_cc, title='Confirmed Cases Autocorrelation')
        plot_pacf(residuals_cc, title='Confirmed Cases Partial Autocorrelation')
        print(fitted_model_cc.summary())

        # fitted_model_rv.plot_predict(arima_order_rv[1], len(country_df_RecoveredCases) + days_num)
        # print(fitted_model_rv.summary())

        fitted_model_fa.plot_predict(arima_order_fa[1], len(country_df_Death) + days_num)
        residuals_fa.plot(title="Death Cases Residuals", legend=None)
        residuals_fa.plot(kind='kde', title='Death Cases Density', legend=None)
        plot_acf(residuals_fa, title='Death Cases Autocorrelation')
        plot_pacf(residuals_fa, title='Death Cases Partial Autocorrelation')
        print(fitted_model_fa.summary())

        plt.show()
    
    return res_ConfirmedCases, res_DeathCases

########################################### " Simulation Set-up" ############################################
show_plots = True
days_num = 90 # two months
country = 'Morocco'
res_ConfirmedCases, res_DeathCases = model_per_country(country, True, days_num)




# # country_df_ConfirmedCases = get_country(confirmed_df, country)
# country_df_Death = get_country(death_df, country)   
# order = (3,2,2)
# rmsle_res = evaluate_arima_model(country_df_Death['Death Cases'].values, order)
# print('>>> Best ARIMA%s RMSLE = %.6f' % (order, rmsle_res))

