#!/usr/bin/env python3
import utils
import scipy

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import numpy as np
import seaborn as sns

from matplotlib.transforms import Bbox
from tqdm import tqdm

from dataclasses import dataclass, field

LABELSIZE = 6


def to_label(xs):
    return ' '.join([str(x) for x in xs])


def to_str(df, groups):
    return df[groups].apply(lambda x: to_label(x), axis=1)


def to_hue(df, groups):
    assert (groups)

    if len(groups) == 1:
        y = df[groups[0]]
    else:
        y = df[groups].to_records(index=False).tolist()

    hue = pd.Series(data=y, index=df.index)
    hue.name = ', '.join(groups)
    return hue


def to_style(df, groups):
    assert (groups)

    if len(groups) > 1:
        groups = groups[0:-1]

    if len(groups) == 1:
        y = df[groups[0]]
    else:
        y = df[groups].to_records(index=False).tolist()

    style = pd.Series(data=y, index=df.index)
    style.name = ', '.join(groups)
    return style


###
# PROCESSING
###


def calculate_similarity_matricies(df, groups, column):
    utils.valid_columns(df, [column])

    d = utils.groupby(df, groups)[column]

    N = len(d)

    KS = np.zeros((N, N)) * np.nan
    MW = np.zeros((N, N)) * np.nan

    labels = []

    for i, (name, x) in tqdm(enumerate(d), desc='calculate_similarity_matricies...', leave=False, total=N):
        if len(groups) == 1:
            name = [name]

        labels.append(to_label(name))

        for j, (_, y) in tqdm(enumerate(d), desc='calculate_similarity_matricies...', leave=False, position=1, total=N):
            if i >= j:
                continue

            res = 1 - scipy.stats.kstest(x, y).pvalue
            KS[i, j] = res
            KS[j, i] = res

            res = 1 - scipy.stats.mannwhitneyu(x, y).pvalue
            MW[i, j] = res
            MW[j, i] = res
    return (pd.DataFrame(data=KS, index=labels, columns=labels), pd.DataFrame(data=MW, index=labels, columns=labels))


###
# LEGEND
###

BACKGROUNDS = {}


def background_changed_callback(evt, ax=None):
    axs = [ax] if ax else evt.canvas.figure.get_axes()

    for ax in axs:
        fig = ax.get_figure()
        # get the active legend
        legend = ax.get_legend()
        if legend:
            # get the bbox around the legend and calculate the bg bbox
            frame = legend.get_tightbbox()
            to_copy = Bbox.from_extents(frame.xmin-10, fig.bbox.ymin-10, frame.xmax+10, fig.bbox.ymax+10)

            # draw everything without the legend
            legend.set_visible(False)
            fig.canvas.draw()

            # copy out the bg
            BACKGROUNDS[ax] = fig.canvas.copy_from_bbox(to_copy)

            # activate the legend again
            legend.set_visible(True)


def legend_scroll_callback(evt):
    ax = evt.inaxes
    if ax and ax in BACKGROUNDS:
        fig = ax.get_figure()
        # get legend
        legend = ax.get_legend()

        # get old frame
        old = legend.get_tightbbox()

        # update
        bbox = legend.get_bbox_to_anchor().transformed(legend.axes.transAxes.inverted())
        legend.set_bbox_to_anchor(Bbox.from_bounds(bbox.x0, bbox.y0+evt.step/25, bbox.width, bbox.height))

        # get new frame
        new = legend.get_tightbbox()

        # compute the area which needs to be updated
        final = Bbox.from_extents(min(new.xmin, old.xmin)-10, min(new.ymin, old.ymin)-10, max(new.xmax, old.xmax)+10, max(new.ymax, old.ymax)+10)

        # restore bg
        fig.canvas.restore_region(BACKGROUNDS[ax])

        # draw legend, update frame buffer and flush
        ax.draw_artist(legend)
        fig.canvas.blit(final)


def add_scrollable_legend(ax, loc='upper right'):
    if not ax.get_legend():
        return

    sns.move_legend(ax, loc=loc)

    fig = ax.get_figure()

    # init once
    background_changed_callback(evt=None, ax=ax)

    def has_callback(name, function):
        if name not in fig.canvas.callbacks.callbacks:
            return False
        return function in map(lambda x: x(), fig.canvas.callbacks.callbacks[name].values())

    if not has_callback('scroll_event', legend_scroll_callback):
        fig.canvas.mpl_connect('scroll_event', legend_scroll_callback)

    if not has_callback('resize_event', background_changed_callback):
        fig.canvas.mpl_connect('resize_event', background_changed_callback)

    ax.callbacks.connect('xlim_changed', lambda ax: background_changed_callback(evt=None, ax=ax))
    ax.callbacks.connect('ylim_changed', lambda ax: background_changed_callback(evt=None, ax=ax))


###
# PLOTTING
###


def create_new_axis(window_title, rows=1, cols=1, sharex=False, sharey=False):
    fig, ax = plt.subplots(rows, cols, sharex=sharex, sharey=sharey)
    fig.canvas.manager.set_window_title(window_title)
    return ax


def plot_samples_line(df, meta, column, xlog=False, xflip=False, in_ax=None):
    utils.valid_columns(df, [column])
    plot_meta = get_plot_meta(df, meta)

    ax = in_ax or create_new_axis(f'Line plot of {column!r}')
    ax = sns.lineplot(data=df, x=df.index, y=column, hue=plot_meta.hue, hue_order=plot_meta.hue_order, style=plot_meta.style,
                      style_order=plot_meta.style_order, ax=ax, palette=plot_meta.palette, legend=True)
    if xlog:
        ax.set(xscale='log')

    if xflip:
        ax.invert_xaxis()

    ax.set_title(f'Lineplot of {column!r}')
    ax.grid(True)

    if not in_ax:
        add_scrollable_legend(ax)


def plot_samples_scatter(df, meta, column, in_ax=None):
    utils.valid_columns(df, [column])
    plot_meta = get_plot_meta(df, meta)

    ax = in_ax or create_new_axis(f'Scatter plot of {column!r}')
    ax = sns.scatterplot(data=df, x=df.index, y=column, hue=plot_meta.hue, hue_order=plot_meta.hue_order, style=plot_meta.style,
                         style_order=plot_meta.style_order, ax=ax, palette=plot_meta.palette, legend=True)

    ax.set_title(f'Scatter plot of {column!r}')
    ax.grid(True)

    vmin = df[column].quantile(0.25)
    vmax = df[column].quantile(0.75)

    ax.axhline(vmin, linestyle='--')
    ax.axhline(vmax, linestyle='--')

    if not in_ax:
        add_scrollable_legend(ax)


def plot_samples_filtered(df, meta, column, in_ax=None, interpolate=True):
    utils.valid_columns(df, [column])
    plot_meta = get_plot_meta(df, meta)

    N = 100

    if interpolate:
        lvl = f'level_{len(meta.groups)}'
        y = df.pivot(columns=meta.groups, values=column)
        scales = y.index.shape[0] / y.count()
        y = y.interpolate(axis=0)
        for c, scale in zip(y.columns, scales):
            y[c] = y[c].rolling(int(N*scale), center=True, min_periods=1).mean()
        y = y.unstack().reset_index(name=column)
        x = lvl
    else:
        y = utils.groupby(df, meta.groups)[column].rolling(N, center=True, min_periods=1).mean().reset_index()
        x = 'index'

    if y[column].isnull().all():
        print('Cannot use moving average ... to few samples!')
        return

    # we need to recompute the hue and style
    hue = to_hue(y, meta.groups)
    style = to_style(y, meta.groups)

    # plot the filtered values
    ax = in_ax or create_new_axis(f'Filtered plot of {column!r}')
    ax = sns.lineplot(data=y, x=x, y=column, hue=hue, hue_order=plot_meta.hue_order, style=style, style_order=plot_meta.style_order, ax=ax, palette=plot_meta.palette, legend=True)

    ax.set_title(f'Filtered Samples of {column!r} {"w" if interpolate else "wo"} interp.')
    ax.set_xlabel('index')
    ax.grid(True)

    if not in_ax:
        add_scrollable_legend(ax)


def plot_samples_histogram(df, meta, column, in_ax=None):
    utils.valid_columns(df, [column])
    plot_meta = get_plot_meta(df, meta)

    ax = in_ax or create_new_axis(f'Histogram plot of {column!r}')
    ax = sns.histplot(data=df, x=column, hue=plot_meta.hue, hue_order=plot_meta.hue_order, ax=ax, element='bars', fill=True, palette=plot_meta.palette, legend=True)

    ax.set_title(f'Histogram plot of {column!r}')
    ax.axvline(df[column].quantile(0.10), linestyle='--')
    ax.axvline(df[column].quantile(0.90), linestyle='--')
    ax.grid(True)

    if not in_ax:
        add_scrollable_legend(ax)


def plot_samples_kde(df, meta, column, in_ax=None):
    utils.valid_columns(df, [column])
    plot_meta = get_plot_meta(df, meta)

    ax = in_ax or create_new_axis(f'KDE plot of {column!r}')
    ax = sns.kdeplot(data=df, x=column, hue=plot_meta.hue, hue_order=plot_meta.hue_order, ax=ax, palette=plot_meta.palette, legend=True)

    ax.set_title(f'KDE plot of {column!r}')
    ax.grid(True)

    if not in_ax:
        add_scrollable_legend(ax)


def plot_similarity_matrix(df, meta, column, name, in_ax=None):
    plot_meta = get_plot_meta(df, meta)

    ax = in_ax or create_new_axis(f'{name} plot of {column!r}')

    ax.set_title(f'{name} plot of {column!r}')
    # ax = sns.heatmap(data=df, vmin=0, vmax=1, annot=False, ax=ax)
    im = ax.imshow(np.array(df), cmap=plot_meta.palette, interpolation='none', aspect='auto', vmin=0, vmax=1)

    ax.get_figure().colorbar(im, ax=ax)

    ax.set_yticks(np.arange(len(df.index)))
    ax.set_xticks(np.arange(len(df.index)))

    ax.set_yticklabels(df.index)
    ax.set_xticklabels(df.index, ha='right')

    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='both', labelsize=LABELSIZE)


def plot_group_stats(df, meta, column, in_axs=None):
    utils.valid_columns(df, [column])

    plot_meta = get_plot_meta(df, meta)

    axs = in_axs or create_new_axis(f'Stats plot of {column!r}', 1, 2)

    df_stats = utils.groupby(df, meta.groups)[column].agg(('count', 'min', 'mean', 'median', 'max', 'std', 'sem')).reset_index()
    print('Stats:')
    print(df_stats)

    for ax, name in zip(axs, ['Mean', 'Median']):

        x = to_str(df_stats, meta.groups)

        ax.set_title(name)
        ax = sns.scatterplot(data=df_stats, x=x, y=name.lower(),  hue=to_hue(df_stats, meta.groups), hue_order=plot_meta.hue_order,
                             style=to_style(df_stats, meta.groups), style_order=plot_meta.style_order, palette=plot_meta.palette, ax=ax, legend=True)

        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='x', labelsize=LABELSIZE)
        ax.set_ylabel('')
        ax.grid(True)

        if not in_axs:
            add_scrollable_legend(ax)


def plot_overview(df, meta, column):

    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title(f'Overview plot of {column!r}')
    plt.subplots_adjust(hspace=0.5)

    gs = gridspec.GridSpec(3, 4)

    axs = [
        plt.subplot(gs[0, 0:2]),
        plt.subplot(gs[0, 2:4]),
        plt.subplot(gs[1, 0:2]),
        plt.subplot(gs[1, 2:4]),
        plt.subplot(gs[2, 0]),
        plt.subplot(gs[2, 1]),
        plt.subplot(gs[2, 2:3]),
        plt.subplot(gs[2, 3:4]),
    ]

    KS, MW = calculate_similarity_matricies(df, meta.groups, column)

    plot_samples_scatter(df, meta, column, axs[0])
    plot_samples_filtered(df, meta, column, axs[1])
    plot_samples_histogram(df, meta, column, axs[2])
    plot_samples_kde(df, meta, column, axs[3])

    plot_similarity_matrix(MW, meta, column, 'MW', axs[4])

    plot_similarity_matrix(KS, meta, column, 'KS', axs[5])

    plot_group_stats(df, meta, column, axs[6:8])

    add_scrollable_legend(axs[0], loc='upper left')
    add_scrollable_legend(axs[1], loc='upper right')
    add_scrollable_legend(axs[2], loc='upper right')
    add_scrollable_legend(axs[3], loc='upper left')
    add_scrollable_legend(axs[6], loc='upper right')
    add_scrollable_legend(axs[7], loc='upper left')

    # axs[1].set_xticklabels(labels=axs[1].get_xticklabels(), ha='right')
    # axs[3].set_xticklabels(labels=axs[3].get_xticklabels(), ha='right')

    # if save:
    #    fig.savefig(f'{file}.{column}_overall.png', dpi=400)
    #    lfig.savefig(f'{file}.{column}_legend.png', dpi=400)

    # import tikzplotlib
    #   tikzplotlib.save(figure=fig,filepath='test.tex')


def plot_heatmap(df, meta, column, X, Y, normalize=False, transpose=False, rotate_labels=False, save=False, axs=None):
    plot_meta = get_plot_meta(df, meta)

    axs = axs or create_new_axis(1, 2)

    stats = ('mean', 'median')

    axs[0].get_figure().suptitle(f'Heatmap {X!r} {Y!r} for {column!r}')

    for i, stat in enumerate(stats):

        x = df.groupby([X, Y]).agg({column: stat}).reset_index()
        x = x.pivot(index=Y, columns=X, values=column)

        if normalize and x.shape[0] != 1:
            x = (x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x))

        if transpose:
            x = x.T

        annot_kws = {'rotation': 'vertical'} if rotate_labels else {}

        # axs[i].set_title(f'{meta.groups}={value} with {str(stat)}')
        sns.heatmap(data=x, annot=True, fmt='.1f', ax=axs[i], cmap=plot_meta.palette, robust=True, annot_kws=annot_kws)


class PlotMeta:

    def __init__(self, df, meta):
        hue = to_hue(df, meta.groups)
        style = to_style(df, meta.groups)

        df[hue.name] = hue
        df[style.name] = style

        self.groups = []
        self.groups.extend(meta.groups)

        self.hue = hue.name
        self.style = style.name

        self.hue_order = hue.unique()
        self.style_order = style.unique()

        self.palette = 'rocket_r'

        try:
            self.hue_order.sort()
        except:
            pass

        try:
            self.style_order.sort()
        except:
            pass


def get_plot_meta(df, meta):
    if 'plot' not in meta.module_data:
        meta.module_data['plot'] = PlotMeta(df, meta)

    if meta.module_data['plot'].groups != meta.groups:
        meta.module_data['plot'] = PlotMeta(df, meta)

    return meta.module_data['plot']
