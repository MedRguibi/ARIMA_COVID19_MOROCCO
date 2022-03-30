import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb ########### pip install lightgbm 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from statsmodels.tsa.arima_model import ARIMA

from time import time

import warnings
warnings.filterwarnings("ignore")


mpl.rcParams['axes.grid']=True
plt.rcParams.update({'figure.figsize':(8, 5), 'figure.dpi':120})

mpl.rcParams['axes.grid']=True
pd.options.display.max_rows = 999

train = pd.read_csv('./Morocco/train.csv')
test = pd.read_csv('./Morocco/test.csv')
submission = pd.read_csv('./Morocco/submission.csv')

train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)


def fill_province(row):
  if pd.isna(row['Province_State']):
    row['Province_State'] = '_PROVINCE_' + row['Country_Region']
  return row

train = train.apply(fill_province, axis = 1)
test = test.apply(fill_province, axis = 1)

def extract_time_features(df):
  df['Day'] = df['Date'].dt.day
  df['Day_of_Week'] = df['Date'].dt.dayofweek
  df['Day_of_Year'] = df['Date'].dt.dayofyear
  df['Week_of_Year'] = df['Date'].dt.weekofyear
  df['Days_im_Month'] = df['Date'].dt.days_in_month


extract_time_features(train)
extract_time_features(test)

train_col_to_delete = ['Id', 'ConfirmedCases', 'Fatalities', 'Country_Region', 'Province_State', 'Date' ]
test_col_to_delete = ['ForecastId', 'Date', 'Country_Region', 'Province_State']
validation_duration = 2
validation_duration = np.timedelta64(validation_duration - 1, 'D')
print(validation_duration)

def train_val_split(df, display = False):
  split_thr = df['Date'].max() - validation_duration 
  df_train = df[df['Date'] < split_thr ]
  X_train = df_train.drop(columns = train_col_to_delete)
  y_cc_train = df_train[['ConfirmedCases']]
  y_fa_train = df_train[['Fatalities']]

  df_val= df[df['Date'] >= split_thr ]
  X_val = df_val.drop(columns = train_col_to_delete)
  y_cc_val = df_val[['ConfirmedCases']]
  y_fa_val = df_val[['Fatalities']]

  if display:
    print('data shape:', df.shape)
    print('train shape:', df_train.shape)
    print('val shape:', df_val.shape)
  return(X_train, y_cc_train, y_fa_train, X_val, y_cc_val, y_fa_val)

def plot_feature_importance(model, X):
    feat_importance = pd.DataFrame(sorted(zip(model.feature_importance(importance_type = 'gain'), X.columns)), columns=['Score','Feature'])
    feat_importance = feat_importance.sort_values(by = "Score", ascending = False)
    plt.figure(figsize = (8, 8))
    sns.barplot(x = "Score", y = "Feature", data = feat_importance)
    plt.title('LightGBM Features')
    plt.tight_layout()
    plt.show()
    return feat_importance.reset_index(drop = True)

def create_model(X_train, y_train, X_val, y_val, draw_metics = False):
  n_estimators = 100
  params = {
  'metric': 'rmse',
  'objective': 'mse',
  'verbose': 0, 
  'learning_rate': 0.99,
  }
  d_train = lgb.Dataset(X_train, y_train)
  d_valid = lgb.Dataset(X_val, y_val)
  watchlist = [d_train, d_valid]
  evals_result = {}
  model = lgb.train(params,
                    d_train, 
                    n_estimators,
                    valid_sets = watchlist, 
                    evals_result = evals_result, 
                    early_stopping_rounds = 10,
                    verbose_eval = 0,
                    )
  if draw_metics:
    lgb.plot_metric(evals_result) 
  return model

def rmse(y, y_hat):
    return np.sqrt(mean_squared_error(y, y_hat))

def rmsle(y, y_hat):
  y_hat = np.where(y_hat < 0, 0, y_hat)
  return 'rmsle', np.sqrt(mean_squared_log_error(y, y_hat))

def evaluate_model(model, X_train, y_train, X_val, y_val): 
    y_hat = model.predict(X_train)
    print('Training error;', rmsle(y_train, y_hat))
    y_val_hat = model.predict(X_val)
    print('Validation error:', rmsle(y_val, y_val_hat))


country = 'Morocco'
province = '_PROVINCE_' + country
country_df = train[(train['Country_Region'] == country) & (train['Province_State'] == province)]

X_train, y_cc_train, y_fa_train, X_val, y_cc_val, y_fa_val = train_val_split(country_df, True)
model_cc = create_model(X_train, y_cc_train, X_val, y_cc_val, True)

plot_feature_importance(model_cc, X_train)

evaluate_model(model_cc, X_train, y_cc_train, X_val, y_cc_val)

# plt.plot(y_cc_train.values)
# plt.plot(model_cc.predict(X_train))

# plt.plot(y_cc_val.values)
# plt.plot(model_cc.predict(X_val))


#Auto Regressor Intergrated Moving average¶
#Define function

def get_country(df, country, province):
    country_df = df[(df['Country_Region'] == country) & (df['Province_State'] == province)]
    country_df = country_df.set_index(keys = 'Date')
    return country_df
def rmsle(y, y_hat):
    y_hat = y_hat.clip(0)
    res = np.sqrt(mean_squared_log_error(y, y_hat))
    return res

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
    print('>>>', target, 'Best ARIMA%s RMSLE = %.3f' % (best_cfg, best_score))
    return best_cfg

p_values = [0, 1, 2, 3]
d_values = range(0, 2)
q_values = range(0, 2)

def model_per_country(country, province, show_plots = False):
    country_df = get_country(train, country, province)
    country_df_test = get_country(test, country, province)

    arima_order_cc = get_arima_order(country_df['ConfirmedCases'].values, p_values, d_values, q_values, 'ConfirmedCases')
    model_cc = ARIMA(country_df.ConfirmedCases, order = arima_order_cc, freq = 'D')
    fitted_model_cc = model_cc.fit(disp = 0)

    arima_order_fa = get_arima_order(country_df['Fatalities'].values, p_values, d_values, q_values, 'Fatalities')
    model_fa = ARIMA(country_df['Fatalities'], order = arima_order_fa, freq = 'D')
    fitted_model_fa = model_fa.fit(disp = 0)
   
    if show_plots:
        fitted_model_cc.plot_predict(arima_order_cc[1], len(country_df) + 90)
        fitted_model_fa.plot_predict(arima_order_fa[1], len(country_df) + 90)
        plt.show()
    return predict_res

show_plots = True
country = 'Morocco'
province = '_PROVINCE_' + country
predict_res = model_per_country(country, province, True)
  