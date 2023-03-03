import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import scipy
import statsmodels.api as sm


print('Hello Anis')

def stats(dframe):
    """Returns statistics of a dataframe."""
    return dframe.agg(['mean', 'var', 'skew', 'kurt', 'min', 'max'])


def log_returns(series):
    """Calculate the log returns of a series."""
    return np.log(series).diff().dropna()


def returns(series):
    """Calculate the simple returns of a series."""
    return series.pct_change().dropna()


def top5(df, n=5):
    """Returns the top n observations"""
    return df.nlargest(n, columns=df[:])


def btm5(df, n=5):
    """Returns the bottom n observations"""
    return df.nsmallest(n, columns=df[:])


def p_norm(data, mean, std):
    """Normalise, and return probability of observations"""
    normalised = (data - mean.values) / std.values
    return pd.DataFrame(data=norm.pdf(normalised), columns=normalised.columns, index=normalised.index)


def kurtosis(df):
    """Calculate the kurtosis of a dataframe."""
    df_kurt = pd.DataFrame(data=scipy.stats.kurtosis(df)).T
    df_kurt.columns = df.columns
    return df_kurt


def skew(df):
    """Calculate the skew of a dataframe."""
    df_skew = pd.DataFrame(data=scipy.stats.skew(df)).T
    df_skew.columns = df.columns
    return df_skew


def jb(data, a):
    """Calculate the Jarque-Bera test of a dataframe."""
    n = data.index.size
    skewness = skew(data)
    kurt = kurtosis(data)
    df_jb = n / 6 * (skewness ** 2 + (kurt ** 2) / 4)
    df_jb = df_jb.T
    df_jb.rename(columns={0: 'JB'}, inplace=True)
    df_jb['p-value'] = 1 - scipy.stats.chi2.cdf(df_jb['JB'], 2)
    df_jb['Check'] = (df_jb['p-value'] < a)
    return df_jb


def ar_stats(data, nlags=10, alpha=0.05, qstat=True):
    """ Calculate the auto-regressive statistics of a dataframe. """

    # Initialize dataframes
    if data is None:
        raise ValueError("Please provide a df")
    else:
        data = pd.DataFrame(data)
        ar_param = pd.DataFrame()

    if alpha is not None:
        conf_int = pd.DataFrame()

    if qstat is not None:
        q_stat = pd.DataFrame()
        p_val = pd.DataFrame()

    # Compute Auto-regressive Statistics
    df_agg_ar_data = data.apply(lambda x: sm.tsa.acf(x, nlags=nlags, alpha=alpha, qstat=qstat))
    df_agg_ar_data = df_agg_ar_data.explode(list(df_agg_ar_data.columns))

    for col in df_agg_ar_data.columns:
        ar_param[col] = df_agg_ar_data[col][0].reset_index(drop=True)
        if alpha is not None:
            conf_int[col] = df_agg_ar_data[col][1].reset_index(drop=True)
        if qstat is not None:
            q_stat[col] = df_agg_ar_data[col][2].reset_index(drop=True)
            p_val[col] = df_agg_ar_data[col][3].reset_index(drop=True)

    return ar_param, conf_int, q_stat, p_val


# Read CSV
df = pd.read_csv('Input/DATA_Project_1.csv', skiprows=1, parse_dates=True, index_col="DATE",
                 infer_datetime_format=True)

# Create Simple Daily Returns
daily_ret = returns(df)
daily_stats = stats(daily_ret)

# Create Daily Log-Returns
daily_log_ret = log_returns(df)
daily_log_stats = stats(daily_log_ret)

# Create Simple Weekly Returns
df_weekly = df.resample('W-FRI').last().copy()
weekly_ret = returns(df_weekly)
weekly_stats = stats(weekly_ret)

# Create Log Weekly Returns
weekly_log_ret = log_returns(df_weekly)
weekly_log_stats = stats(weekly_log_ret)


# 1a Daily Mean Returns Difference
diff_daily_mean_ret = (daily_log_ret - daily_ret).mean()

# 1b Weekly Mean Returns Difference
diff_weekly_mean_ret = (weekly_log_ret - weekly_ret).mean()

# 1c Difference Between Log and Simple on Weekly and Daily Timeframes
diff_log_sim_mean_ret = (daily_log_ret - weekly_log_ret).mean()

# 2a.
# Daily
top5_log_daily = top5(daily_log_ret, 5)
btm5_log_daily = btm5(daily_log_ret, 5)

# Weekly
top5_log_weekly = top5(weekly_log_ret, 5)
btm5_log_weekly = btm5(weekly_log_ret, 5)

# 2b.
# Daily
daily_std = pd.DataFrame(daily_log_ret.std()).T
daily_mean = pd.DataFrame(daily_log_ret.mean()).T

# Weekly
weekly_std = pd.DataFrame(weekly_log_ret.std()).T
weekly_mean = pd.DataFrame(weekly_log_ret.mean()).T

# Probability of observing daily returns under a normal distribution
top5_log_daily_prob = p_norm(data=top5_log_daily, mean=daily_mean, std=daily_std)
btm5_log_daily_prob = p_norm(data=btm5_log_daily, mean=daily_mean, std=daily_std)

# Probability of observing weekly returns under a normal distribution
top5_log_weekly_prob = p_norm(data=top5_log_weekly, mean=weekly_mean, std=weekly_std)
btm5_log_weekly_prob = p_norm(data=btm5_log_weekly, mean=weekly_mean, std=weekly_std)

# 2c.
# Daily Returns Kurtosis
daily_log_kurt = kurtosis(daily_log_ret)
daily_log_skew = skew(daily_log_ret)

# Weekly Returns Kurtosis
weekly_log_kurt = kurtosis(weekly_log_ret)
weekly_log_skew = skew(weekly_log_ret)

# JB Statistic
# Daily
jb_daily = jb(daily_log_ret, a=0.05)

# Weekly
jb_weekly = jb(weekly_log_ret, a=0.05)

# 2d
# Daily
d_acf, d_cint, d_qstat, d_pval = ar_stats(daily_log_ret, alpha=0.05, qstat=True)
d_log_test = d_pval.applymap(lambda x: np.where(x < 0.05, 'Reject', 'Accept'))

# Weekly
w_acf, w_cint, w_qstat, w_pval = ar_stats(weekly_log_ret, alpha=0.05, qstat=True)
w_log_test = w_pval.applymap(lambda x: np.where(x < 0.05, 'Reject', 'Accept'))

# Daily squared returns
daily_sq_ret = daily_log_ret ** 2
d_sq_acf, d_sq_cint, d_sq_qstat, d_sq_pval = ar_stats(daily_sq_ret, qstat=True)
d_sq_test = d_sq_pval.applymap(lambda x: np.where(x < 0.05, 'Reject', 'Accept'))

# Weekly squared returns
weekly_sq_ret = weekly_log_ret ** 2
w_sq_acf, w_sq_cint, w_sq_qstat, w_sq_pval = ar_stats(weekly_sq_ret, qstat=True)
w_sq_test = w_sq_pval.applymap(lambda x: np.where(x < 0.05, 'Reject', 'Accept'))

# 2e
# Interpretation

# 3a
# Construct equally weighted portfolio returns
# Daily
daily_port_ret = pd.DataFrame(data=daily_ret.mean(axis=1).dropna(), columns=['DailyPortRet'])
daily_port_stats = stats(daily_port_ret)

# 3b
# Weekly
weekly_port_ret = pd.DataFrame(data=weekly_ret.mean(axis=1).dropna(), columns=['WeeklyPortRet'])
weekly_port_stats = stats(weekly_port_ret)
