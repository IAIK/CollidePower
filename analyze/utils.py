def split_comma(ctx, params, value):
    return value.split(',') if value else []


def valid_columns(df, columns):
    for c in columns:
        if c not in df:
            print()
            print(f'unknown column: {c}')
            cols = ','.join(map(repr, df.columns))
            print(f'available columns: {cols}')
            exit(-1)


def get_named_argument(function, name, args, kwargs):
    import inspect
    index = inspect.getargspec(function).args.index(name)
    if index < len(args):
        return args[index]
    else:
        return kwargs[name]


def groupby(df, groups):
    if groups:
        return df.groupby(groups, group_keys=False, as_index=False)
    else:
        return df.groupby(['_'], group_keys=False, as_index=False)


class color_fg:
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    orange = '\033[33m'
    blue = '\033[34m'
    purple = '\033[35m'
    cyan = '\033[36m'
    lightgrey = '\033[37m'
    darkgrey = '\033[90m'
    lightred = '\033[91m'
    lightgreen = '\033[92m'
    yellow = '\033[93m'
    lightblue = '\033[94m'
    pink = '\033[95m'
    lightcyan = '\033[96m'
    end = '\033[0m'


class color_bg:
    black = '\033[40m'
    red = '\033[41m'
    green = '\033[42m'
    orange = '\033[43m'
    blue = '\033[44m'
    purple = '\033[45m'
    cyan = '\033[46m'
    lightgrey = '\033[47m'
    end = '\033[0m'
