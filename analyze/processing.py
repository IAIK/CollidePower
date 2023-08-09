import utils
from functools import wraps


def report_dropped_samples(function):
    def wrapper(df, *args, **kwargs):
        n = df.index.shape[0]
        df = function(df, *args, **kwargs)
        n_removed = n - df.index.shape[0]
        per = n_removed/n*100 if n != 0 else 100
        # print(f"removed {n_removed:10d} samples ({per:5.1f}%)")
        return df
    return wrapper


def foreach_column_masked(name):
    def decorator(function):
        @wraps(function)
        def wrapper(df, *args, **kwargs):
            def ww(*args, **kwargs):
                return utils.get_named_argument(function, name, args, kwargs)
            columns = ww(df, *args, **kwargs)
            utils.valid_columns(df, columns)
            mask = function(df[columns[0]], *args, **kwargs)
            for c in columns[1:]:
                mask &= function(df[c], *args, **kwargs)
            return df[mask]
        return wrapper
    return decorator


def moving_filter(df, columns, filter_type, N):
    utils.valid_columns(df, columns)

    for c in columns:
        if filter_type == 'mean':
            df[c] -= df[c].rolling(N, center=True, min_periods=1).mean()
        elif filter_type == 'median':
            df[c] -= df[c].rolling(N, center=True, min_periods=1).median()
        else:
            print(f'unknown filter: {filter_type}')
            exit(-1)
    return df


def run_code(df, code):
    if df.index.shape[0] == 0:
        return df

    print(f'Running {code!r} ', end='')

    local_vars = {'x': df}

    while True:
        try:
            exec(code, local_vars)
            break
        except NameError as e:
            var = str(e).split('\'')[1]  # todo fix with python 3.10
            print(f' converting variable {var} to string ', end='')
            local_vars[var] = var
        except Exception as e:
            print(f'\ncode {code!r} raised exception: {e}')
            exit(-1)

    # for c in df.columns:
    #    if pd.api.types.is_categorical_dtype(df[c]):
    #        df[c] = df[c].cat.remove_unused_categories()

    return local_vars['x']


@ report_dropped_samples
def select(df, code):
    return run_code(df, f'x=x[{code}]')


def run(df, code):
    return run_code(df, code)


def remove_outliers(x, lower, upper):
    if lower < upper:
        return (x >= lower) & (x <= upper)
    else:
        return (x >= lower) | (x <= upper)


@ report_dropped_samples
@ foreach_column_masked('outliers')
def remove_outliers_percentile(x, outliers, lower, upper):
    l = x.quantile(lower/100)
    u = x.quantile(upper/100)
    return remove_outliers(x, l, u)


@ report_dropped_samples
@ foreach_column_masked('outliers')
def remove_outliers_standard(x, outliers, lower, upper):
    l = x.mean() + lower * x.std()
    u = x.mean() + upper * x.std()
    return remove_outliers(x, l, u)


@ report_dropped_samples
@ foreach_column_masked('outliers')
def remove_outliers_cut(x, outliers, lower, upper):
    return remove_outliers(x, lower, upper)


@ report_dropped_samples
def slice_trace(df, begin, end):
    n = df.index.shape[0]
    b = int(begin / 100.0 * n)
    e = int(end / 100.0 * n)
    return df.iloc[b:e]


@ report_dropped_samples
@ foreach_column_masked('outliers')
def remove_outliers_moving_percentile(x, outliers, lower, upper, N):
    u = x.rolling(N, center=True, min_periods=1).quantile(upper/100)
    l = x.rolling(N, center=True, min_periods=1).quantile(lower/100)
    return (x >= l) & (x <= u)


@ report_dropped_samples
@ foreach_column_masked('outliers')
def remove_outliers_moving_std(x, outliers, lower, upper, N):
    s = x.rolling(N, center=True, min_periods=1).std()
    m = x.rolling(N, center=True, min_periods=1).mean()
    l = m + lower * s
    u = m + upper * s
    return (x >= l) & (x <= u)
