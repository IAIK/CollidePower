import utils
import click

from dataclasses import dataclass
from functools import wraps


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'], ignore_unknown_options=True, allow_extra_args=True)


def format_dict(x, key_color='', value_color=''):
    return ', '.join([f'{key_color}{k}{utils.color_fg.end}={value_color}{v!r}{utils.color_fg.end}' for k, v in x.items()])


def header(ctx, click_kwargs, newline=True):
    s = format_dict(click_kwargs, key_color=utils.color_fg.lightcyan)
    print(f'{utils.color_fg.cyan}>{utils.color_fg.end} {utils.color_fg.green}{ctx.command.name}{utils.color_fg.end} {s}\t', flush=True, end=None if newline else ' ')


def group_header(df, meta, newline=True):
    if df is None or not meta.is_grouped():
        return
    v = format_dict(dict(zip(meta.groups, meta.group_value(df))), key_color=utils.color_fg.orange)
    print(f'{utils.color_fg.cyan}>>{utils.color_fg.end} {v}\t', flush=True, end=None if newline else ' ')


@dataclass(frozen=True)
class Meta:
    groups = ['_']
    saved_groups = {}
    saved_dfs = {}
    module_data = {}

    def save(self, dfs, id):
        self.saved_groups[id] = self.groups.copy()
        self.saved_dfs[id] = dfs

    def load(self, id):
        self.groups.clear()
        self.groups.extend(self.saved_groups[id])
        return self.raw_load(id)

    def raw_load(self, id):
        return self.saved_dfs[id]

    def group_value(self, df):
        return tuple(df[self.groups].iloc[0],)

    def is_grouped(self):
        return self.groups != ['_']

    def group_header(self, df, newline=True):
        group_header(df, self, newline)


def do_grouping(dfs, meta, groups):
    from warnings import catch_warnings, simplefilter

    if len(dfs) != 1:
        print('can only group once, use \'-u\' to ungroup first, or specify multiple columns with \'-g <column1>,<column2>\'')
        exit(-1)

    df = dfs[0]

    utils.valid_columns(df, groups)

    with catch_warnings():
        simplefilter(action='ignore', category=FutureWarning)
        if df.index.shape[0] == 0:
            print('error empty data frame cannot be grouped')
            exit(0)

        _, value_dfs = zip(*df.groupby(groups, group_keys=True, as_index=False))

    meta.groups.clear()
    meta.groups.extend(groups)

    return list(value_dfs)


def do_ungrouping(dfs, meta):
    from pandas import concat
    meta.groups.clear()
    meta.groups.extend(['_'])
    return [concat(dfs)]


def do_merging(meta, id_column, ids):
    from pandas import concat

    def load_dfs(id):
        df = concat(meta.raw_load(id))  # ungroup just to make sure
        df[id_column] = id
        return df

    dfs = [load_dfs(id) for id in ids]

    return do_ungrouping(dfs, meta)


def remove_empty(dfs):

    non_empty = [x for x in dfs if x is not None and x.index.shape[0]]
    for df in non_empty:
        if '_' not in df:
            df['_'] = 1

    return non_empty if len(non_empty) else [None]


def process_pipeline(processors):
    dfs = [None]
    meta = Meta()

    for p in processors:
        dfs = remove_empty(dfs)
        dfs = p(dfs, meta)

    from matplotlib.pyplot import show
    show()


def pipeline(newline=True):
    def decorator(function):
        @ wraps(function)
        def click_wrapper(ctx, **click_kwargs):

            @ wraps(function)
            def pipeline_wrapper(dfs, meta):

                header(ctx, click_kwargs, newline=meta.is_grouped() or newline)

                def invoke(df):
                    # group_header(df, meta, newline=newline)
                    return function(df=df, meta=meta, **click_kwargs)

                return list(map(invoke, dfs))

            return pipeline_wrapper
        return click.pass_context(click_wrapper)
    return decorator


def pipeline_ungrouped(newline=True):
    def decorator(function):
        @ wraps(function)
        def click_wrapper(ctx, **click_kwargs):
            @ wraps(function)
            def pipeline_wrapper(dfs, meta):
                header(ctx, click_kwargs, newline=newline)
                return function(dfs=dfs, meta=meta, **click_kwargs)
            return pipeline_wrapper
        return click.pass_context(click_wrapper)
    return decorator


def group():
    return click.group(chain=True, invoke_without_command=True, result_callback=process_pipeline, context_settings=CONTEXT_SETTINGS)


def add_command(group, name, short_help):

    def wrapper(function):
        c = group.command(name=name, context_settings=CONTEXT_SETTINGS)(function)
        usage = c.get_usage(click.Context(c))
        help_text = ('> ' + short_help) if short_help else ''
        c.short_help = f'{usage:70} {help_text}'
        return c

    return wrapper


def command(group, name, short_help=None):
    return add_command(group, '--' + name, short_help)


def shortcommand(group, name, short_help=None):
    return add_command(group, '-' + name, short_help)


##
# Auto/Shell completion
##

@ group()
def cli():
    pass


##
# Loading and Saving Data
##


@ shortcommand(cli, 'i', short_help='load csv file')
@ click.argument('file_name', type=click.Path(exists=True, dir_okay=False))
@ pipeline_ungrouped()
def cmd_open(dfs, meta, file_name):
    from pandas import read_csv
    df = read_csv(file_name, comment='#', memory_map=True)
    df['_'] = 1
    print(f'loaded file {file_name!r} with {df.index.shape[0]} samples')
    return do_ungrouping([df], meta)


@ shortcommand(cli, 'o', short_help='store csv file')
@ click.argument('file_name', type=click.Path(dir_okay=False))
@ pipeline_ungrouped()
def cmd_store(dfs, meta, file_name):
    from pandas import concat
    concat(dfs).to_csv(file_name)

    return dfs


##
# Grouping and Ungrouping Data
##


@ shortcommand(cli, 'g', short_help='group data')
@ click.argument('columns', type=click.STRING, callback=utils.split_comma)
@ pipeline_ungrouped()
def cmd_group(dfs, meta, columns):
    return do_grouping(dfs, meta, list(columns))


@ shortcommand(cli, 'u', short_help='ungroup data')
@ pipeline_ungrouped()
def cmd_ungroup(dfs, meta):
    return do_ungrouping(dfs, meta)


##
# Intermediate Results
##

@ shortcommand(cli, 's', short_help='save intermediate data')
@ click.argument('id', type=click.STRING)
@ pipeline_ungrouped()
def cmd_push(dfs, meta, id):
    meta.save(dfs, id)
    return dfs


@ shortcommand(cli, 'l', short_help='load intermediate data')
@ click.argument('id', type=click.STRING)
@ pipeline_ungrouped()
def cmd_use(dfs, meta, id):
    return meta.load(id)


@ command(cli, 'merge', short_help='load intermediate data')
@ click.argument('id_column', type=click.STRING)
@ click.argument('ids', type=click.STRING, callback=utils.split_comma)
@ pipeline_ungrouped()
def cmd_merge(dfs, meta, id_column, ids):
    return do_merging(meta, id_column, ids)


@ command(cli, 'print', short_help='print current data frame')
@ pipeline_ungrouped()
def cmd_print(dfs, meta):
    from pandas import concat, option_context
    with option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 160):
        print(concat(dfs))
    return dfs


@ command(cli, 'csv', short_help='print csv of current data frame')
@ pipeline_ungrouped()
def cmd_print(dfs, meta):
    from pandas import concat, option_context
    with option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 160):
        print(concat(dfs).to_csv())
    return dfs


@ command(cli, 'head', short_help='head of current data frame')
@ pipeline_ungrouped()
def cmd_print(dfs, meta):
    from pandas import concat
    x = concat(dfs)
    print(x.columns)
    print(x)
    return dfs


##
# Plotting
##


@ command(cli, 'plot.overview')
@ click.argument('column', type=click.STRING)
@ pipeline_ungrouped()
def cmd_plot_overview(dfs, meta, column):
    from plot import plot_overview
    from pandas import concat
    plot_overview(concat(dfs).sort_index(), meta, column)
    return dfs


@ command(cli, 'plot.line')
@ click.argument('column', type=click.STRING)
@ click.argument('xlog', type=click.BOOL)
@ pipeline_ungrouped()
def cmd_plot_line(dfs, meta, column, xlog):
    from plot import plot_samples_line
    from pandas import concat
    plot_samples_line(concat(dfs).sort_index(), meta, column, xlog=xlog)
    return dfs


@ command(cli, 'plot.line_xf')
@ click.argument('column', type=click.STRING)
@ click.argument('xlog', type=click.BOOL)
@ pipeline_ungrouped()
def cmd_plot_line_xflipped(dfs, meta, column, xlog):
    from plot import plot_samples_line
    from pandas import concat
    plot_samples_line(concat(dfs).sort_index(), meta, column, xlog=xlog, xflip=True)
    return dfs


@ command(cli, 'plot.scatter')
@ click.argument('column', type=click.STRING)
@ pipeline_ungrouped()
def cmd_plot_scatter(dfs, meta, column):
    from plot import plot_samples_scatter
    from pandas import concat
    plot_samples_scatter(concat(dfs).sort_index(), meta, column)
    return dfs


@ command(cli, 'plot.filtered')
@ click.argument('column', type=click.STRING)
@ click.argument('interpolate', type=click.BOOL)
@ pipeline_ungrouped()
def cmd_plot_filtered(dfs, meta, column, interpolate):
    from plot import plot_samples_filtered
    from pandas import concat
    plot_samples_filtered(concat(dfs).sort_index(), meta, column, interpolate=interpolate)
    return dfs


@ command(cli, 'plot.hist')
@ click.argument('column', type=click.STRING)
@ pipeline_ungrouped()
def cmd_plot_hist(dfs, meta, column):
    from plot import plot_samples_histogram
    from pandas import concat
    plot_samples_histogram(concat(dfs).sort_index(), meta, column)
    return dfs


@ command(cli, 'plot.kde')
@ click.argument('column', type=click.STRING)
@ pipeline_ungrouped()
def cmd_plot_kde(dfs, meta, column):
    from plot import plot_samples_kde
    from pandas import concat
    plot_samples_kde(concat(dfs).sort_index(), meta, column)
    return dfs


@ command(cli, 'plot.stats')
@ click.argument('column', type=click.STRING)
@ pipeline_ungrouped()
def cmd_plot_stats(dfs, meta, column):
    from plot import plot_group_stats
    from pandas import concat
    plot_group_stats(concat(dfs).sort_index(), meta, column)
    return dfs


@ command(cli, 'plot.heatmap')
@ click.argument('column', type=click.STRING)
@ click.argument('X', type=click.STRING)
@ click.argument('Y', type=click.STRING)
@ pipeline_ungrouped()
def cmd_plot_heatmap(dfs, meta, column, x, y):
    from plot import plot_heatmap
    for df in dfs:
        plot_heatmap(df, meta, column, x, y)
    return dfs


@ command(cli, 'plot.show')
@ pipeline_ungrouped()
def cmd_plot_show(dfs, meta):
    from matplotlib.pyplot import show
    show()
    return dfs


##
# Processing
##

@ command(cli, 'filter', short_help='subtract moving average filter')
@ click.argument('columns', type=click.STRING, callback=utils.split_comma)
@ click.argument('filter_type', type=click.Choice(['mean', 'median']))
@ click.argument('number_samples', type=click.INT, default=100)
@ pipeline()
def cmd_filter(df, meta, columns, filter_type, number_samples):
    from processing import moving_filter
    return moving_filter(df, columns, filter_type, number_samples)


@ command(cli, 'sel', short_help='select subset of data')
@ click.argument('code', type=click.STRING)
@ pipeline()
def cmd_select(df, meta, code):
    from processing import select
    return select(df, code)


@ command(cli, 'run', short_help='run code snippet')
@ click.argument('code', type=click.STRING)
@ pipeline()
def cmd_run(df, meta, code):
    from processing import run
    return run(df, code)


@ command(cli, 'idx', short_help='set data frame index')
@ click.argument('columns', type=click.STRING, callback=utils.split_comma)
@ pipeline()
def cmd_select(df, meta, columns):
    utils.valid_columns(df, columns)
    return df.set_index(columns)


@ command(cli, 'per', short_help='remove percentile outliers')
@ click.argument('outliers', type=click.STRING, callback=utils.split_comma)
@ click.argument('lower', type=click.FloatRange(0, 100))
@ click.argument('upper', type=click.FloatRange(0, 100))
@ pipeline(newline=False)
def cmd_per(df, meta, outliers, lower, upper):
    from processing import remove_outliers_percentile
    return remove_outliers_percentile(df, outliers, lower, upper)


@ command(cli, 'std', short_help='remove standard outliers')
@ click.argument('outliers', type=click.STRING, callback=utils.split_comma)
@ click.argument('lower', type=click.FLOAT)
@ click.argument('upper', type=click.FLOAT)
@ pipeline(newline=False)
def cmd_std(df, meta, outliers, lower, upper):
    from processing import remove_outliers_standard
    return remove_outliers_standard(df, outliers, lower, upper)


@ command(cli, 'cut', short_help='cutaway outliers')
@ click.argument('outliers', type=click.STRING, callback=utils.split_comma)
@ click.argument('lower', type=click.FLOAT)
@ click.argument('upper', type=click.FLOAT)
@ pipeline(newline=False)
def cmd_cut(df, meta, outliers, lower, upper):
    from processing import remove_outliers_cut
    return remove_outliers_cut(df, outliers, lower, upper)


@ command(cli, 'mvper', short_help='cutaway outliers')
@ click.argument('outliers', type=click.STRING, callback=utils.split_comma)
@ click.argument('lower', type=click.FLOAT)
@ click.argument('upper', type=click.FLOAT)
@ click.argument('number_samples', type=click.INT, default=100)
@ pipeline(newline=False)
def cmd_cut(df, meta, outliers, lower, upper, number_samples):
    from processing import remove_outliers_moving_percentile
    return remove_outliers_moving_percentile(df, outliers, lower, upper, number_samples)


@ command(cli, 'mvstd', short_help='cutaway outliers')
@ click.argument('outliers', type=click.STRING, callback=utils.split_comma)
@ click.argument('lower', type=click.FLOAT)
@ click.argument('upper', type=click.FLOAT)
@ click.argument('number_samples', type=click.INT, default=100)
@ pipeline(newline=False)
def cmd_cut(df, meta, outliers, lower, upper, number_samples):
    from processing import remove_outliers_moving_std
    return remove_outliers_moving_std(df, outliers, lower, upper, number_samples)


##
# Power
##


@ command(cli, 'power.init')
@ click.argument('scale', type=click.STRING, default="")
@ click.argument('explode', type=click.BOOL)
@ pipeline()
def cmd_power_init(df, meta, scale, explode):
    from power import power_init_df
    return power_init_df(df, meta, scale, explode)


@ command(cli, 'power.set_functions', short_help='set the statistic functions to use')
@ click.argument('cor_function', type=click.Choice(['pearson', 'spearman']))
@ click.argument('lr_function', type=click.Choice(['classic', 'huber']))
@ pipeline_ungrouped()
def cmd_power_set_functions(dfs, meta, cor_function, lr_function):
    from power import power_set_functions
    power_set_functions(meta, cor_function, lr_function)
    return dfs


@ command(cli, 'power.cpa_edge_cases', short_help='set of the CPA should also perform edge cases')
@ click.argument('do_edge_cases', type=click.BOOL)
@ pipeline_ungrouped()
def cmd_power_set_functions(dfs, meta, do_edge_cases):
    from power import power_exclude_edge_cases
    power_exclude_edge_cases(meta, not do_edge_cases)
    return dfs


@ command(cli, 'power.set_comp', short_help='set the model components')
@ click.argument('components', type=click.STRING, callback=utils.split_comma)
@ pipeline_ungrouped()
def cmd_power_model(dfs, meta, components):
    from power import power_set_model_components
    power_set_model_components(meta, components)
    return dfs


@ command(cli, 'power.set_coefs', short_help='set the model coefficients')
@ click.argument('coefficients', type=click.STRING, callback=utils.split_comma)
@ pipeline_ungrouped()
def cmd_power_model(dfs, meta, coefficients):
    from power import power_set_model_coefficients
    power_set_model_coefficients(meta, list(map(float, coefficients)))
    return dfs


@ command(cli, 'power.find_coefs', short_help='find the model coefficients for the model components')
@ click.argument('column', type=click.STRING)
@ click.argument('independent_coefficients', type=click.BOOL)
@ pipeline_ungrouped()
def cmd_power_model(dfs, meta, column, independent_coefficients):
    from power import power_find_model_coefficients
    return power_find_model_coefficients(dfs, meta, column, independent_coefficients=independent_coefficients)


@ command(cli, 'power.cpa', short_help='perform a CPA on the data column')
@ click.argument('column', type=click.STRING)
@ click.argument('repetitions', type=click.INT)
@ click.argument('n_samples', type=click.STRING, callback=utils.split_comma)
@ pipeline_ungrouped()
def cmd_power_cpa(dfs, meta, column, repetitions, n_samples):
    from power import power_cpa
    return power_cpa(dfs, meta, column, repetitions=repetitions, n_samples_to_test=list(map(int, n_samples)))


def main():
    cli()


if __name__ == '__main__':
    main()
