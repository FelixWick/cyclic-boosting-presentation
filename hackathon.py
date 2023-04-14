import sys

import pandas as pd
import numpy as np
import datetime

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

from cyclic_boosting import binning, flags, CBPoissonRegressor, observers, common_smoothers
from cyclic_boosting.smoothing.onedim import SeasonalSmoother, IsotonicRegressor
from cyclic_boosting.plots import plot_analysis

from IPython import embed


def plot_CB(filename, plobs, binner):
    for i, p in enumerate(plobs):
        plot_analysis(
            plot_observer=p,
            file_obj=filename + "_{}".format(i), use_tightlayout=False,
            binners=[binner]
        )


def eval_results(yhat_mean, y):
    mad = np.nanmean(np.abs(y - yhat_mean))
    print('MAD: {}'.format(mad))
    mse = np.nanmean(np.square(y - yhat_mean))
    print('MSE: {}'.format(mse))
    mape = np.nansum(np.abs(y - yhat_mean)) / np.nansum(y)
    print('MAPE: {}'.format(mape))
    smape = 100. * np.nanmean(np.abs(y - yhat_mean) / ((np.abs(y) + np.abs(yhat_mean)) / 2.))
    print('SMAPE: {}'.format(smape))
    md = np.nanmean(y - yhat_mean)
    print('MD: {}'.format(md))

    mean_y = np.nanmean(y)
    print('mean(y): {}'.format(mean_y))


def get_events(df):
    for event in [
        'Christmas',
        'Easter',
        'Labour_Day',
        'German_Unity',
        'Other_Holiday',
        'Local_Holiday_0',
        'Local_Holiday_1',
        'Local_Holiday_2'
    ]:
        for event_date in df['DATE'][df['EVENT'] == event].unique():
            for event_days in range(-10, 11):
                df.loc[df['DATE'] == pd.to_datetime(event_date) + datetime.timedelta(days=event_days), event] = event_days

    return df


def prepare_data(df):
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['dayofweek'] = df['DATE'].dt.dayofweek
    df['dayofyear'] = df['DATE'].dt.dayofyear
    df['month'] = df['DATE'].dt.month
    df['dayofmonth'] = df['DATE'].dt.day

    df['td'] = (df['DATE'] - df['DATE'].min()).dt.days

    df['price_ratio'] = df['SALES_PRICE'] / df['NORMAL_PRICE']
    df['price_ratio'].fillna(1, inplace=True)
    df['price_ratio'].clip(0, 1, inplace=True)
    df.loc[df['price_ratio'] == 1., 'price_ratio'] = np.nan

    df = get_events(df)

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    df[['L_ID', 'P_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3']] = enc.fit_transform(df[['L_ID', 'P_ID', 'PG_ID_1', 'PG_ID_2', 'PG_ID_3']])

    return df


def fill_gaps(df):
    df_dates = pd.DataFrame(
        {
            "DATE": pd.date_range(start=df['DATE'].min(), end=df['DATE'].max()),
        }
    )

    df = df_dates.merge(df, on="DATE", how="left")

    df['SALES'].fillna(0, inplace=True)
    defaults = {
        'PG_ID_1': df['PG_ID_1'].iloc[0],
        'PG_ID_2': df['PG_ID_2'].iloc[0],
        'PG_ID_3': df['PG_ID_3'].iloc[0],
        'NORMAL_PRICE': df['NORMAL_PRICE'].iloc[0],
        'SALES_AREA': df['SALES_AREA'].iloc[0],
        'SCHOOL_HOLIDAY': 0.0,
        'PROMOTION_TYPE': 0.0,
        'SALES_PRICE': df['NORMAL_PRICE'].iloc[0]
    }
    df.fillna(value=defaults, inplace=True)

    return df


def fill_zeros(df):
    df = df.groupby(['L_ID', 'P_ID']).apply(fill_gaps)
    df = df.drop(columns=['L_ID', 'P_ID']).reset_index()

    return df


def feature_properties():
    fp = {}
    fp['P_ID'] = flags.IS_UNORDERED
    fp['PG_ID_1'] = flags.IS_UNORDERED
    fp['PG_ID_2'] = flags.IS_UNORDERED
    fp['PG_ID_3'] = flags.IS_UNORDERED
    fp['L_ID'] = flags.IS_UNORDERED
    fp['dayofweek'] = flags.IS_ORDERED
    fp['month'] = flags.IS_ORDERED
    fp['dayofyear'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    fp['dayofmonth'] = flags.IS_CONTINUOUS
    fp['price_ratio'] = flags.IS_CONTINUOUS | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['PROMOTION_TYPE'] = flags.IS_ORDERED
    fp['SCHOOL_HOLIDAY'] = flags.IS_ORDERED
    fp['Christmas'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Easter'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Labour_Day'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['German_Unity'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Other_Holiday'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Local_Holiday_0'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Local_Holiday_1'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['Local_Holiday_2'] = flags.IS_ORDERED  | flags.HAS_MISSING | flags.MISSING_NOT_LEARNED
    fp['NORMAL_PRICE'] = flags.IS_CONTINUOUS
    fp['td'] = flags.IS_CONTINUOUS | flags.IS_LINEAR
    return fp


def cb_model():
    fp = feature_properties()
    explicit_smoothers = {('dayofyear',): SeasonalSmoother(order=3),
                          ('price_ratio',): IsotonicRegressor(increasing=False),
                          ('NORMAL_PRICE',): IsotonicRegressor(increasing=False),
                         }

    features = [
        'dayofweek',
        'L_ID',
        'PG_ID_1',
        'PG_ID_2',
        'PG_ID_3',
        'P_ID',
        'PROMOTION_TYPE',
        'price_ratio',
        'dayofyear',
        'month',
        'dayofmonth',
        'SCHOOL_HOLIDAY',
        'Christmas',
        'Easter',
        'Labour_Day',
        'German_Unity',
        'Other_Holiday',
        'Local_Holiday_0',
        'Local_Holiday_1',
        'Local_Holiday_2',
        ('L_ID', 'td'),
        ('P_ID', 'td'),
        ('P_ID', 'L_ID'),
        ('L_ID', 'dayofweek'),
        ('PG_ID_1', 'dayofweek'),
        ('PG_ID_2', 'dayofweek'),
        ('PG_ID_3', 'dayofweek'),
        ('P_ID', 'dayofweek'),
        ('L_ID', 'PG_ID_1', 'dayofweek'),
        ('L_ID', 'PG_ID_2', 'dayofweek'),
        ('L_ID', 'PG_ID_3', 'dayofweek'),
        ('SCHOOL_HOLIDAY', 'dayofweek'),
        ('SCHOOL_HOLIDAY', 'L_ID', 'dayofweek'),
        ('SCHOOL_HOLIDAY', 'PG_ID_3', 'dayofweek'),
        ('SCHOOL_HOLIDAY', 'L_ID', 'PG_ID_3', 'dayofweek'),
        ('L_ID', 'dayofmonth'),
        ('PG_ID_3', 'dayofmonth'),
        ('L_ID', 'PG_ID_3', 'dayofmonth'),
        ('L_ID', 'dayofyear'),
        ('PG_ID_3', 'dayofyear'),
        ('P_ID', 'dayofyear'),
        ('L_ID', 'PG_ID_3', 'dayofyear'),
        ('L_ID', 'Christmas'),
        ('L_ID', 'Easter'),
        ('L_ID', 'Labour_Day'),
        ('L_ID', 'German_Unity'),
        ('L_ID', 'Local_Holiday_0'),
        ('L_ID', 'Local_Holiday_1'),
        ('PG_ID_3', 'Christmas'),
        ('PG_ID_3', 'Easter'),
        ('PG_ID_3', 'Labour_Day'),
        ('PG_ID_3', 'German_Unity'),
        ('PG_ID_3', 'Local_Holiday_0'),
        ('PG_ID_3', 'Local_Holiday_1'),
        ('P_ID', 'Christmas'),
        ('P_ID', 'Easter'),
        ('P_ID', 'Labour_Day'),
        ('P_ID', 'German_Unity'),
        ('P_ID', 'Local_Holiday_0'),
        ('P_ID', 'Local_Holiday_1'),
        ('L_ID', 'PG_ID_3', 'Christmas'),
        ('L_ID', 'PG_ID_3', 'Easter'),
        ('L_ID', 'PG_ID_3', 'Labour_Day'),
        ('L_ID', 'PG_ID_3', 'German_Unity'),
        ('L_ID', 'PG_ID_3', 'Local_Holiday_0'),
        ('L_ID', 'PG_ID_3', 'Local_Holiday_1'),
        ('PROMOTION_TYPE', 'dayofweek'),
        ('price_ratio', 'dayofweek'),
        ('P_ID', 'PROMOTION_TYPE'),
        ('P_ID', 'price_ratio'),
        'NORMAL_PRICE',
    ]

    plobs = [observers.PlottingObserver(iteration=1), observers.PlottingObserver(iteration=-1)]

    est = CBPoissonRegressor(
        feature_properties=fp,
        feature_groups=features,
        observers=plobs,
        maximal_iterations=50,
        smoother_choice=common_smoothers.SmootherChoiceGroupBy(
            use_regression_type=True,
            use_normalization=False,
            explicit_smoothers=explicit_smoothers),
    )

    binner = binning.BinNumberTransformer(n_bins=100, feature_properties=fp)

    ml_est = Pipeline([("binning", binner), ("CB", est)])
    return ml_est


def training(X, y):
    CB_est = cb_model()
    CB_est.fit(X, y)

    plot_CB('analysis_CB_mean_iterlast', [CB_est[-1].observers[0], CB_est[-1].observers[-1]], CB_est[-2])

    del X
    return CB_est


def inference(X, ml_est_mean):
    yhat_mean = ml_est_mean.predict(X)

    del X
    return yhat_mean


def main(args):
    df_train = pd.read_parquet("blueyonder-pyconpydata-2023/train_BY_hackathon_final.parquet.gzip")
    df_test = pd.read_parquet("blueyonder-pyconpydata-2023/test_BY_hackathon_without_sales_final.parquet.gzip")

    # fill zeros
    df_train = fill_zeros(df_train)

    df_test['SALES'] = np.nan

    df = pd.concat([df_train, df_test], ignore_index=True)

    df = prepare_data(df)

    df_train = df.loc[df['DATE']<='2022-03-31']
    df_test = df.loc[df['DATE']>'2022-03-31']

    # cut out anomalies
    df_train = df_train.loc[df_train['SALES'] >= 0]
    df_train = df_train.loc[df_train['SALES'] < 1000]

    y_train = np.asarray(df_train['SALES'])
    X_train = df_train.drop(columns='SALES')
    X_test = df_test.drop(columns='SALES')

    CB_est = training(X_train.copy(), y_train)

    X_train['yhat'] = inference(X_train.copy(), CB_est)
    # in-sample evaluation
    X_train['y'] = y_train
    eval_results(X_train['yhat'], X_train['y'])

    X_test['Predicted'] = inference(X_test.copy(), CB_est)

    X_test = X_test[['Id', 'Predicted']]
    X_test.to_csv("CB_master_submission.csv", index=False)

    # out-of-sample evaluation
    X_test.reset_index(drop=True, inplace=True)
    df_y_test = pd.read_parquet("blueyonder-pyconpydata-2023/test_BY_hackathon_results_final.parquet.gzip")
    df_y_test.reset_index(drop=True, inplace=True)
    df_y_test['Predicted'] = X_test['Predicted']
    eval_results(df_y_test['Predicted'], df_y_test['Expected'])
    eval_results(df_y_test.loc[df_y_test['Usage'] == 'Public', 'Predicted'], df_y_test.loc[df_y_test['Usage'] == 'Public', 'Expected'])
    eval_results(df_y_test.loc[df_y_test['Usage'] == 'Private', 'Predicted'], df_y_test.loc[df_y_test['Usage'] == 'Private', 'Expected'])

    embed()


if __name__ == "__main__":
    main(sys.argv[1:])
