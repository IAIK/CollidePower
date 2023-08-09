import utils

import pandas as pd
import numpy as np
import multiprocessing as mp

from shared_buffer import SharedNumpyArray

from scipy.stats import pearsonr, spearmanr, kendalltau
from hashlib import sha512
from re import match

from tqdm import tqdm
from typing import List, Union
from warnings import catch_warnings, simplefilter
from dataclasses import dataclass, field


from sklearn.linear_model import LinearRegression, HuberRegressor, RANSACRegressor


@dataclass(frozen=True)
class Word:
    '''Models a data word'''
    name: str
    index: int

    def eval(self, vs, gs):
        assert (self.name in ['v', 'g'])
        if self.name == 'v':
            return vs[:, self.index]
        else:
            return gs[:, self.index]

    def __str__(self):
        return f'{self.name}{self.index}'

    @staticmethod
    def from_string(x):
        if m := match('([vg])(\d+)', x):
            return Word(m[1], int(m[2]))
        return None


@dataclass(frozen=True)
class HW:
    '''Model the hamming weight of a single word'''
    x: Word

    def __str__(self):
        return f'hw_{self.x}'

    def eval(self, vs, gs):
        return hw(self.x.eval(vs, gs))

    @staticmethod
    def from_string(x):
        if m := match('hw_(\w+)', x):
            if w := Word.from_string(m[1]):
                return HW(w)
        return None


@dataclass(frozen=True)
class HD:
    '''Model the hamming distance between two words'''
    x: Word
    y: Word

    def eval(self, vs, gs):
        return hd(self.x.eval(vs, gs), self.y.eval(vs, gs))

    def __str__(self):
        return f'hd_{self.x}_{self.y}'

    @staticmethod
    def from_string(x):
        if m := match('hd_(\w+)_(\w+)', x):
            if x := Word.from_string(m[1]):
                if y := Word.from_string(m[2]):
                    return HD(x, y)
        return None


@dataclass
class PowerMeta:
    '''class to keep track of the power related meta data'''

    # Defines the sizes of an analysis word
    word_bits: int = 0
    word_count: int = 0
    word_dtype = np.uint8

    expander = None
    scaled = False

    # Victim and Guess words
    vs: List[Word] = field(default_factory=list)
    gs: List[Word] = field(default_factory=list)

    # string rep for pandas
    vs_str: List[str] = field(default_factory=list)
    gs_str: List[str] = field(default_factory=list)

    # the model components
    model_components: List[Union[HW, HD]] = field(default_factory=list)
    model_coefficients: List[float] = field(default_factory=list)


assert (HW.from_string('hw_v0') != HW.from_string('hw_g0'))
assert (HW.from_string('hw_v0') != HW.from_string('hw_v1'))
assert (HW.from_string('hw_v0') == HW.from_string('hw_v0'))
assert (hash(HW.from_string('hw_v0')) != hash(HW.from_string('hw_g0')))
assert (hash(HW.from_string('hw_v0')) != hash(HW.from_string('hw_v1')))
assert (hash(HW.from_string('hw_v0')) == hash(HW.from_string('hw_v0')))

assert (HD.from_string('hd_v0_g0') != HD.from_string('hd_g0_g0'))
assert (HD.from_string('hd_v0_g0') != HD.from_string('hd_v1_g0'))
assert (HD.from_string('hd_v0_g0') == HD.from_string('hd_v0_g0'))
assert (hash(HD.from_string('hd_v0_g0')) != hash(HD.from_string('hd_g0_g0')))
assert (hash(HD.from_string('hd_v0_g0')) != hash(HD.from_string('hd_v1_g0')))
assert (hash(HD.from_string('hd_v0_g0')) == hash(HD.from_string('hd_v0_g0')))


def hw(arr):
    '''vectorized hamming weight computation'''
    # taken from
    # https://stackoverflow.com/questions/63954102/numpy-vectorized-way-to-count-non-zero-bits-in-array-of-integers
    # outperforms other approaches by orders of magnitude

    t = arr.dtype.type
    mask = np.iinfo(t).max
    s55 = t(0x5555555555555555 & mask)
    s33 = t(0x3333333333333333 & mask)
    s0F = t(0x0F0F0F0F0F0F0F0F & mask)
    s01 = t(0x0101010101010101 & mask)

    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)
    arr = (arr + (arr >> 4)) & s0F
    return ((arr * s01) >> (8 * (arr.itemsize - 1))).astype(np.uint8)


def hd(x, y):
    '''vectorized hamming distance computation'''
    return hw(np.bitwise_xor(x, y))


class SimpleExpander:
    def __init__(self, dtype, word_count):
        self.dtype = dtype
        self.word_count = word_count

    def __call__(self, x):
        n = x.shape[0]
        return x.view(self.dtype).reshape(n, -1)[:, :self.word_count]
        # return x.view(self.dtype)[:self.word_count]


class ExtendedExpander:
    def __init__(self, word_bits, word_count):
        self.word_count = word_count
        self.shift = np.array(range(0, 8, word_bits), dtype=np.uint8)
        self.mask = (np.array((1 << word_bits) - 1, dtype=np.uint8).reshape((-1, 1)) << self.shift)[0]

    def __call__(self, x):
        n = x.shape[0]
        return ((x.view(np.uint8).reshape((-1, 1)) & self.mask) >> self.shift).reshape(n, -1)[:, :self.word_count]
        # return ((x.view(np.uint8).reshape((-1, 1)) & self.mask) >> self.shift).reshape((1, -1))[0, :self.word_count]


def create_expander(word_bits, word_count):
    '''create expander to transform uint8 stream to the desired word_bits width and word_count '''
    # check if the expander yields a 'nice' data type
    d = {
        8: np.uint8,
        16: np.uint16,
        32: np.uint32,
        64: np.uint64,
    }
    if word_bits in d:
        return SimpleExpander(d[word_bits], word_count)

    if word_bits not in [1, 2, 4]:
        raise 'not implemented!'

    return ExtendedExpander(word_bits, word_count)


assert (np.all(create_expander(2, 8)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[0, 2, 0, 0, 3, 0, 3, 0]])))
assert (np.all(create_expander(2, 6)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[0, 2, 0, 0, 3, 0]])))
assert (np.all(create_expander(4, 4)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[8, 0, 3, 3]])))
assert (np.all(create_expander(8, 2)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[8, 0x33]])))
assert (np.all(create_expander(16, 1)(np.array([[0x8, 0x33]], dtype=np.uint8)) == np.array([[0x3308]])))


def compute_hypothesis(vs, gs, model):
    '''compute the hypothesis based on the given victim, guess data, and the model'''
    return sum([component.eval(vs, gs) * coefficient for component, coefficient in model.items()])


def compute_hypotheses(
        vs_candidates: np.array,
        gs: np.array,
        model
):
    '''compute all hypothesis for each vs candidate'''
    H = np.zeros((vs_candidates.shape[0], gs.shape[0]))

    for i, vs in enumerate(vs_candidates):
        v = np.broadcast_to(vs, (gs.shape[0], vs.size))
        H[i, :] = compute_hypothesis(v, gs, model)

    return H


def compute_lr(meta, x, y):
    '''compute linear regression'''

    if meta.lr_function == 'classic':
        lr = LinearRegression(positive=True, fit_intercept=True)
    else:
        lr = HuberRegressor()

    return lr.fit(x, y)


def compute_rho(meta, x, y):
    '''compute correlation coefficient, confidence interval and pvalue'''

    if meta.cor_function == 'pearson':
        result = pearsonr(x, y)
        data = [result.statistic] + list(map('{:5.2f}'.format, (*result.confidence_interval(), result.pvalue)))
    else:
        result = spearmanr(x, y)
        data = [result.correlation]

    return dict(zip(meta.cor_stats, data))


def compute_snr(vs, gs, y, model):
    '''compute the SNR between the aligned components and the rest of the signal'''

    def is_aligned(item):
        component, _ = item
        return isinstance(component, HD) and component.x.index == component.y.index

    model_aligned = dict(filter(is_aligned, model.items()))

    signal = compute_hypothesis(vs, gs, model_aligned)
    noise = y - signal

    return np.var(signal) / np.var(noise)


def aligned_model(meta: PowerMeta):
    return [HD(v, g) for v, g in zip(meta.vs, meta.gs)]


def cross_model(meta: PowerMeta):
    return [HD(v, g) for v in meta.vs for g in meta.gs if v.index != g.index]


def self_model(meta: PowerMeta):
    return [HD(v1, v2) for v1 in meta.vs for v2 in meta.vs if v1.index > v2.index] + [HD(g1, g2) for g1 in meta.gs for g2 in meta.gs if g1.index > g2.index]


def guess_weight_model(meta: PowerMeta):
    return [HW(g) for g in meta.gs]


def victim_weight_model(meta: PowerMeta):
    return [HW(v) for v in meta.vs]


def power_set_model_components(meta, components):
    '''set the model components, like HD and HW'''

    pmeta = meta.module_data['power']

    mcs = []

    for c in components:
        if x := HW.from_string(c):
            mcs.append(x)
        elif x := HD.from_string(c):
            mcs.append(x)
        elif c == 'default':
            mcs.extend(aligned_model(pmeta))
            mcs.extend(guess_weight_model(pmeta))
        elif c == 'aligned':
            mcs.extend(aligned_model(pmeta))
        elif c == 'self':
            mcs.extend(self_model(pmeta))
        elif c == 'cross':
            mcs.extend(cross_model(pmeta))
        elif c == 'guess_w':
            mcs.extend(guess_weight_model(pmeta))
        elif c == 'victim_w':
            mcs.extend(victim_weight_model(pmeta))
        elif c == 'all':
            mcs.extend(aligned_model(pmeta))
            mcs.extend(self_model(pmeta))
            mcs.extend(cross_model(pmeta))
            mcs.extend(guess_weight_model(pmeta))
            mcs.extend(victim_weight_model(pmeta))
        else:
            print(f'unknown model component {c}')
            exit(-1)

    print('model components: {}'.format(', '.join(map(str, mcs))))

    meta.module_data['power'].model_components = mcs


def power_set_model_coefficients(meta, coefficients):
    '''set the model coefficients, for the given model components'''
    meta.module_data['power'].model_coefficients = coefficients


def power_set_functions(meta, cor_function, lr_function):
    '''set the statistical function for the correlation coefficients and the linear regression'''

    assert (cor_function in ['pearson', 'spearman'])
    assert (lr_function in ['classic', 'huber'])

    print(f'using {lr_function!r} as linear regression method')
    print(f'using {cor_function!r} as correlation coefficient')

    meta.module_data['power'].cor_function = cor_function
    meta.module_data['power'].lr_function = lr_function

    if cor_function == 'pearson':
        meta.module_data['power'].cor_stats = ['rho', 'rho_l', 'rho_u', 'pv']
    else:
        meta.module_data['power'].cor_stats = ['rho']


def power_exclude_edge_cases(meta, exclude_edge_cases):
    meta.module_data['power'].exclude_edge_cases = exclude_edge_cases
    s = '' if exclude_edge_cases else 'NOT '
    print(f'{s}excluding edge cases')

##
# Init Power
##


def power_init_df(df, meta, scale, explode):
    '''init the data frame for additional analysis'''
    if 'Value' not in df or 'Guess' not in df:
        print('data frame does not have columns "Value"/"Guess"')
        exit(-1)

    pmeta = PowerMeta()

    df = add_compatibility_mode(df)

    # we are using the sha mode if we have more than 16-bit values
    is_guess_sha = df.Guess.max() > np.iinfo(np.uint16).max

    # only if the guess is in sha mode, we extend the words
    if is_guess_sha:
        mode = 'SHA'
        pmeta.word_bits = 64
        pmeta.word_count = 512 // pmeta.word_bits
        pmeta.word_dtype = np.uint64
    else:
        is_distinct = 'CLVFill' in df and (df['CLVFill'] == 'D').any() or 'CLGFill' in df and (df['CLGFill'] == 'D').any()

        if is_distinct:
            mode = 'DISTINCT'
        else:
            mode = 'NORMAL'

        pmeta.word_bits = 4
        pmeta.word_count = 2 if is_distinct else 1
        pmeta.word_dtype = np.uint8

    print(f'{mode}-mode using {pmeta.word_count} {pmeta.word_bits}-bit WORDS')

    pmeta.expander = create_expander(pmeta.word_bits, pmeta.word_count)

    # df = save_space(df)

    df = fix_overflows(df)
    df = add_scale(df, pmeta, scale)
    df = do_expand(df, pmeta)

    if explode:
        df = do_explode(df, pmeta)

    df = add_power(df)
    df = add_diff(df)

    meta.module_data['power'] = pmeta

    power_set_model_components(meta, ['default'])

    power_set_functions(meta, 'pearson', 'classic')
    power_exclude_edge_cases(meta, False)

    return df


def save_space(df):

    # Wed Nov 30 06:52:43 2022,
    # %a %b %d %H:%M:%S %Y
    # df['hours'] = pd.to_datetime(df.TimeStamp, format='%a %b %d %H:%M:%S %Y')
    # df['hours'] = (df['hours'] - df['hours'].iloc[0]) / np.timedelta64(1, 'h')
    # df = df.set_index('dt')

    before = df.memory_usage(deep=True).sum()

    to_drop = [x for x in ['TimeStamp', 'RAPerf', 'RMperf', 'RCalib', 'IAPerf', 'IMperf', 'ICalib'] if x in df]
    df = df.drop(to_drop, axis=1)

    for c in df.columns:
        if pd.api.types.is_object_dtype(df.dtypes[c]):
            df = df.astype({c: 'category'})

    # zero_columns = df.columns[(df == 0).all()]
    # for c in zero_columns:
    #    df = df.astype({c: 'category'})

    after = df.memory_usage(deep=True).sum()

    print(f'reduced data frame size from {before/10**6:.1f} MB to {after/10**6:.1f} MB')

    return df


def add_compatibility_mode(df):
    if 'IEnergy' not in df:
        df['IEnergy'] = df.REnergy - df.DEnergy
        df['ITicks'] = df.RTicks - df.DTicks

    if 'CLVFill' not in df and 'CLFill' in df:
        df['CLVFill'] = df['CLFill']

    if 'CLGFill' not in df and 'CLFill' in df:
        df['CLGFill'] = df['CLFill']

    return df


def fix_overflows(df):
    for x in ['REnergy', 'IEnergy', 'REnergyPP0', 'IEnergyPP0']:
        if x in df:
            df = df.astype({x: np.uint32})
            df = df.astype({x: np.int64})

    return df


def norm(x):
    return (x - x.mean()) / x.std()


def add_power(df):
    df['RPower'] = df.REnergy / df.RTicks
    df['IPower'] = df.IEnergy / df.ITicks

    df['RFreq'] = df.RAPerf / df.RMperf
    df['IFreq'] = df.IAPerf / df.IMperf

    if 'REnergyPP0' in df and 'IEnergyPP0' in df:
        df['RPowerPP0'] = df.REnergyPP0 / df.RTicks
        df['IPowerPP0'] = df.IEnergyPP0 / df.ITicks

        df['RDPower'] = df.RPower - df.RPowerPP0
        df['IDPower'] = df.IPower - df.IPowerPP0

    df['RNPower'] = norm(df.REnergy) + norm(df.RTicks)
    df['INPower'] = norm(df.IEnergy) + norm(df.ITicks)

    return df


def add_diff(df):
    df['DTicks'] = df.RTicks - df.ITicks

    df['DEnergy'] = df.REnergy - df.IEnergy
    df['DPower'] = df.RPower - df.IPower

    df['DNPower'] = df.RNPower - df.INPower

    # this metric is useful if the system is already thermal throttling
    df['DNEnergyTicks'] = (df.DEnergy-df.DEnergy.mean())/df.DEnergy.std()+(df.DTicks-df.DTicks.mean())/df.DTicks.std()

    if 'REnergyPP0' in df and 'IEnergyPP0' in df:
        df['DEnergyPP0'] = df.REnergyPP0 - df.IEnergyPP0
        df['DPowerPP0'] = df.RPowerPP0 - df.IPowerPP0

        # this metric is useful if the system is already thermal throttling
        df['DNEnergyTicksPP0'] = (df.DEnergyPP0-df.DEnergyPP0.mean())/df.DEnergyPP0.std()+(df.DTicks-df.DTicks.mean())/df.DTicks.std()

        df['DDPower'] = df.RDPower - df.IDPower

    return df


def add_scale(df, pmeta, scale):

    from unit_scaling import time_unit_s, energy_unit_j

    column_scales = {
        'mlab07': {'Energy': 6.103515625e-05, 'EnergyPP0': 6.103515625e-05, 'Ticks': 2.7027027e-10},
        'ulab07': {'Ticks': 4.1666667e-10},
        'lab10': {'Ticks': 2.7777778e-10, 'Energy': 6.103515625e-05},
        'lab06': {'Ticks': 2.5e-10},
        'config': {'Energy': energy_unit_j, 'EnergyPP0': energy_unit_j, 'Ticks': time_unit_s}
    }

    if scale not in column_scales:
        print(f'No unit scaling!')
        return df

    print(f'Using unit scaling: {column_scales[scale]}')

    for column, scale in column_scales[scale].items():
        for pre in ['R', 'I']:
            c = pre + column
            if c in df:
                df[c] *= scale

    pmeta.scaled = True

    print('number of seconds per Real sample:')
    print(df.groupby('Exp').RTicks.mean())

    return df


def cl_from_hash(x):
    # expand the value to a 64 byte hash
    x_hash = sha512(np.uint64(x)).digest()
    # transform to uint8 view
    return np.frombuffer(x_hash, dtype=np.uint8)


def do_expand(df, meta: PowerMeta):
    print('expanding')

    # in the distinct case we add the fills to the values
    if 'VFill' in df:
        df['Value'] = df['Value'] + np.left_shift(df['VFill'], meta.word_bits)

    if 'GFill' in df:
        df['Guess'] = df['Guess'] + np.left_shift(df['GFill'], meta.word_bits)

    # the words available in the data frame
    meta.vs = [Word('v', i) for i in range(meta.word_count)]
    meta.gs = [Word('g', i) for i in range(meta.word_count)]

    # pandas likes string columns
    meta.vs_str = list(map(str, meta.vs))
    meta.gs_str = list(map(str, meta.gs))

    def expand(df, c, t, r):
        mask = (df[t] == 'R')

        if mask.any():
            df.loc[mask, r] = meta.expander(df.loc[mask].apply(lambda x: cl_from_hash(x[c]), axis=1, result_type='expand').values)

        mask = ~mask
        if mask.any():
            df.loc[mask, r] = meta.expander(df.loc[mask, c].values)

        df[r] = df[r].astype(meta.word_dtype)

        return df

    df = expand(df, 'Value', 'CLVFill', meta.vs_str)
    df = expand(df, 'Guess', 'CLGFill', meta.gs_str)

    hw_v = hw(df[meta.vs_str].values)
    hw_g = hw(df[meta.gs_str].values)

    df['hw_v'] = hw_v.sum(axis=1)
    df['hw_g'] = hw_g.sum(axis=1)

    # save the words hamming weights for convenience
    df[list(map(lambda x: str(HW(x)), meta.vs))]=hw_v
    df[list(map(lambda x: str(HW(x)), meta.gs))]=hw_g

    df['hw_vg'] = df.hw_v + df.hw_g

    # calculate the hamming distances
    df['hd'] = hd(df[meta.vs_str].values, df[meta.gs_str].values).sum(axis=1)

    df = df.drop(['Value', 'Guess'], axis=1)

    return df


def do_explode(df, meta: PowerMeta):
    print('exploding')
    df_copy = df.copy()

    d = {}
    for c in ['Energy', 'EnergyPP0', 'Ticks', 'Volt', 'Temp']:
        d.update({'R' + c: 'I' + c, 'I' + c: 'R' + c})

    df_copy = df_copy.rename(columns=d)

    invert_mask = (1 << meta.word_bits) - 1

    df_copy[meta.gs_str] = df_copy[meta.gs_str] ^ invert_mask
    return pd.concat([df, df_copy], ignore_index=True)


##
# Find Model Coefficients
##


def power_find_model_coefficients(dfs, meta, column, independent_coefficients):
    '''find the model coefficients using linear regression'''

    pmeta = meta.module_data['power']

    # if independent_coefficients:
    #    df_cor = pd.DataFrame()
    # else:
    result = []
    for i, df in enumerate(dfs):
        utils.valid_columns(df, [column])

        # find the coefficients and bring them into pandas format
        model, rho, snr = find_model_coefficients_linear_regression(df, pmeta, column, independent_coefficients)
        d = dict(zip(map(str, model.keys()), model.values()))
        d.update(rho)
        d['N'] = df.index.shape[0]
        d['SNR'] = snr * 1000

        # if independent_coefficients:
        #    df_cor = pd.DataFrame()
        #    for mc, c in model.items():
        #        if isinstance(mc, HD):
        #            name = 'hd_' + mc.x.name + mc.y.name
        #            df_cor = df_cor.append({'x': str(mc.x), 'y': str(mc.y), name: c}, ignore_index=True)
        #        if isinstance(mc, HW):
        #            name = 'hw_' + mc.x.name
        #            df_cor = df_cor.append({'x': str(mc.x), name: c}, ignore_index=True)
        #    print(df_cor)
        #    return df_cor
        # else:

        tmp = pd.DataFrame(d, columns=pmeta.cor_stats + list(map(str, pmeta.model_components)) + ['N', 'SNR'], index=[i])
        tmp[meta.groups] = meta.group_value(df)

        tmp[list(map(str, pmeta.model_components))] *= 1  # 1000/128

        result.append(tmp)

    print(pd.concat(result, ignore_index=True).set_index(meta.groups))
    return result


def find_model_coefficients_linear_regression(
    df: pd.DataFrame,
    meta: PowerMeta,
    y_column: str,
    independent_coefficients: bool
):

    # get the used Words
    vs = df[meta.vs_str].to_numpy()
    gs = df[meta.gs_str].to_numpy()
    # and the measured signal
    y = df[y_column].to_numpy()

    if not independent_coefficients:

        model_components_str = list(map(str, meta.model_components))

        x = pd.DataFrame(columns=model_components_str, index=df.index, dtype=np.uint16)

        # compute each element of the hypothesis
        for mc in meta.model_components:
            x[str(mc)] = mc.eval(vs, gs)

        # compute the linear regression and compute the model coefficients

        result = compute_lr(meta, x[model_components_str], y)
        model = dict(zip(meta.model_components, result.coef_))

    else:
        # share the data
        vs_s = SharedNumpyArray(vs)
        gs_s = SharedNumpyArray(gs)
        y_s = SharedNumpyArray(y)

        with mp.Pool(mp.cpu_count()) as pool:

            args = [(meta, vs_s, gs_s, y_s, mc) for mc in meta.model_components]

            # compute each model coefficients independently
            results = pool.imap_unordered(independent_linear_regression, args)

            model = {}
            for r in tqdm(results, total=len(meta.model_components), desc='LR', leave=False):
                model.update(r)

        vs_s.unlink()
        gs_s.unlink()
        y_s.unlink()

    # compute rho and snr based on the computed model
    rho = compute_rho(meta, compute_hypothesis(vs, gs, model), y)
    snr = compute_snr(vs, gs, y, model)

    return model, rho, snr


def independent_linear_regression(arg):
    meta, vs_s, gs_s, y_s, mc = arg
    x = mc.eval(vs_s.read(), gs_s.read()).reshape(-1, 1)
    result = compute_lr(meta, x, y_s.read())
    return {mc: result.coef_[0]}


##
# CPA
##

def power_cpa(
        dfs: List[pd.DataFrame],
        meta: list,
        column: str,
        repetitions: int = 10,
        n_samples_to_test: List[int] = [0]
):

    pmeta = meta.module_data['power']

    # compute the search space
    word_bits_overall = pmeta.word_bits * pmeta.word_count

    # this is already much
    assert (word_bits_overall <= 10)

    # generate the CPA candidates based on the word configuration
    vs_candidates = pmeta.expander(np.arange(1 << word_bits_overall))
    if pmeta.exclude_edge_cases:
        vs_candidates = vs_candidates[1:-1]  # uncomment to remove 0 and -1
    vs_candidates = SharedNumpyArray(vs_candidates)
    results = []

    # crate the processing pool used for sampling here
    with mp.Pool() as pool:
        # perform the cpa for each of the grouped data frames
        for df in dfs:
            results.append(power_analysis_cpa_one_df(
                df=df,
                meta=meta,
                column=column,
                vs_candidates=vs_candidates,
                repetitions=repetitions,
                n_samples_to_test=n_samples_to_test,
                pool=pool
            ))

    # unlink shared memory
    vs_candidates.unlink()

    return results


def power_analysis_cpa_one_df(
        df: pd.DataFrame,
        meta,
        column: str,
        vs_candidates: SharedNumpyArray,
        repetitions: int,
        n_samples_to_test: List[int],
        pool
):
    utils.valid_columns(df, [column])
    meta.group_header(df)
    pmeta = meta.module_data['power']

    if pmeta.scaled:
        seconds_per_sample = df.RTicks.mean()

    # create result data frame format
    result = pd.DataFrame()

    # get the coefficients for the CPA
    if pmeta.model_coefficients:
        if len(pmeta.model_components) != len(pmeta.model_coefficients):
            print('missing coefficients!')
            exit(-1)
        model = dict(zip(pmeta.model_components, pmeta.model_coefficients))
        rho = np.NaN
    else:
        model, rho, _ = find_model_coefficients_linear_regression(df, pmeta, column, independent_coefficients=False)

    # debug
    print('using model: (rho={}) {}'.format(rho, ', '.join([f'{mc}={c}' for mc, c in model.items()])))

    # get the threshold for the edge detection
    threshold, threshold_column = get_edge_case_threshold(df, column)

    # run over all the sample lengths to test
    for n_samples in n_samples_to_test:

        print(f'n_samples={n_samples}')

        # we measure how often the correct value is within the first 3 candidates
        cpa_performance = np.zeros((3,))

        # ignore the pandas warning for groupby tuples
        with catch_warnings():
            simplefilter(action='ignore', category=FutureWarning)

            value_groups = df.groupby(pmeta.vs_str)

            # perform the cpa for each constant victim value in the df
            for vs_ground_truth, df_vs in value_groups:

                vs_ground_truth = np.array(vs_ground_truth if isinstance(vs_ground_truth, tuple) else (vs_ground_truth,))

                cpa_performance += power_analysis_cpa_one_value(
                    df=df_vs,
                    meta=pmeta,
                    column=column,
                    vs_candidates=vs_candidates,
                    vs_ground_truth=vs_ground_truth,
                    model=model,
                    threshold_column=threshold_column,
                    threshold=threshold,
                    repetitions=repetitions,
                    n_samples=n_samples,
                    pool=pool
                )

        cpa_performance /= len(value_groups)

        res = dict(zip(['1st', '2nd', '3rd'], cpa_performance))
        res.update({'n_tries': n_samples})

        if pmeta.scaled:
            minutes = n_samples * seconds_per_sample / 60
            bit_per_minute = 4 / minutes

            if column.startswith('D'):
                # if we use a differential measurement we need 2 samples
                bit_per_minute = bit_per_minute / 2

            res.update({'bit_per_minute': bit_per_minute})
            res.update({'bit_per_hour': bit_per_minute * 60})

        result = pd.concat([result, pd.DataFrame(res, index=[0])], ignore_index=True)

        out = '% '.join(map('{0:5.1f}'.format, cpa_performance*100))
        print(f'AVG  = {out}%')

    result[meta.groups] = meta.group_value(df)
    result = result.set_index(['n_tries'])
    print(result)
    return result


def power_analysis_cpa_one_value(
        df: pd.DataFrame,
        meta: PowerMeta,
        column: str,
        vs_candidates: SharedNumpyArray,
        vs_ground_truth,
        model,
        threshold_column: str,
        threshold: float,
        repetitions: int,
        n_samples: int,
        pool
):
    gs = df[meta.gs_str].to_numpy()
    y = df[column].to_numpy()
    y_threshold = df[threshold_column].to_numpy()
    H = compute_hypotheses(vs_candidates.read(), gs, model)

    # share the data frame columns
    y_s = SharedNumpyArray(y)
    y_threshold_s = SharedNumpyArray(y_threshold)
    H_s = SharedNumpyArray(H)

    args = [(meta, vs_candidates, H_s, y_s, y_threshold_s, n_samples, vs_ground_truth, threshold, n_samples == 0)]

    # check if we should use all available samples or sample multiple result
    if n_samples == 0:
        cpa_performance, cpa_result, cpa_scores, corrected_str = sampled_cpa(args[0])
        # generate output string
        out = ' '.join([print_candidate(vs_ground_truth, c, s) for c, s in zip(cpa_result[:5], cpa_scores[:5])]) + ' ' + corrected_str
    else:
        results = pool.imap_unordered(sampled_cpa, args*repetitions)

        cpa_performance = sum([r[0] for r in tqdm(results, total=repetitions, desc='CPA', leave=False)]) / repetitions
        # generate output string
        out = ' '.join(map('{0:5.1f}%'.format, cpa_performance*100))

    print(f'N={df.index.shape[0]:5}, V={format_value(vs_ground_truth)} => {out}')

    # unlink the shared columns
    H_s.unlink()
    y_s.unlink()
    y_threshold_s.unlink()

    return cpa_performance


def sampled_cpa(args):
    '''perform the CPA with randomly selected samples'''

    meta, vs_candidates, H, y, y_threshold, n_samples, vs_ground_truth, threshold, norm = args

    # read the shared arrays
    vs_candidates = vs_candidates.read()
    H = H.read()
    y = y.read()
    y_threshold = y_threshold.read()

    n_max = H.shape[1]

    # randomly select samples
    # if n_samples is zero -> all samples
    rng = np.random.default_rng()
    sample_size = n_max if n_samples == 0 else min(n_samples, n_max)
    select = rng.choice(n_max, size=sample_size, replace=False)

    # get the cpa result
    cpa_result, cpa_scores = compute_cpa_index(meta, vs_candidates, H, y, select, norm)

    # correct the edge cases
    cpa_result, corrected_str = correct_edge_cases(vs_candidates, y_threshold, cpa_result, threshold, select)

    # analyze the results and return the score
    return cpa_performance(vs_ground_truth, cpa_result), cpa_result, cpa_scores, corrected_str


def compute_cpa_index(
        meta: PowerMeta,
        vs_candidates: np.array,
        H: np.array,
        y: np.array,
        select: np.array,
        norm=True
):

    if meta.cor_function == 'pearson':
        # this is way faster
        cpa_result = np.corrcoef(H[:, select], y[select])[-1, :-1]
    else:
        # cpa based on the guessed victim value
        cpa_result = np.array([compute_rho(meta, H[i, select], y[select])['rho'] for i, _ in enumerate(vs_candidates)])

    # norm the cpa result
    if norm and not np.all(np.isnan(cpa_result)):
        cpa_result = (cpa_result - np.nanmin(cpa_result)) / (np.nanmax(cpa_result) - np.nanmin(cpa_result)) * 100

    indices = np.argsort(cpa_result)[::-1]

    # sort based on the score
    vs_candidates = vs_candidates[indices]
    cpa_result = cpa_result[indices]

    # move NANs to the back
    mask = ~np.isnan(cpa_result)
    vs_candidates = np.concatenate((vs_candidates[mask], vs_candidates[~mask]))
    cpa_result = np.concatenate((cpa_result[mask], cpa_result[~mask]))

    return vs_candidates, cpa_result


def correct_edge_cases(vs_candidates, y, cpa_result, threshold, select):

    corrected_str = ''

    if threshold is None:
        return cpa_result, corrected_str

    if (all(cpa_result[0] == vs_candidates[0]) or all(cpa_result[0] == vs_candidates[-1])) and (all(cpa_result[1] == vs_candidates[0]) or all(cpa_result[1] == vs_candidates[-1])):
        m = np.mean(y[select])

        if m < threshold:
            cpa_result[0], cpa_result[1] = vs_candidates[0], vs_candidates[-1]
            sym = '<'
        else:
            cpa_result[0], cpa_result[1] = vs_candidates[-1], vs_candidates[0]
            sym = '>='

        corrected_str = f'found edge case: mean={m} {sym} threshold={threshold} -> {cpa_result[0]}'

    return cpa_result, corrected_str


def cpa_performance(ground_truth, cpa_result):
    '''compute how good the CPA result was, i.e, check if the ground truth is in the first, second, or third candidate'''
    # breakpoint()

    bit_error_rate = False

    if bit_error_rate:
        return np.array([hd(np.array(ground_truth), cpa_result[0, :])/4, hd(np.array(ground_truth), cpa_result[1, :])/4, hd(np.array(ground_truth), cpa_result[2, :])/4])[:, 0]
    else:
        return np.array([np.any(np.all(ground_truth == cpa_result[0:i+1, :], axis=1)) for i in range(3)])


def format_value(value):
    return np.array2string(value, formatter={'all': '{:2}'.format})


def print_candidate(real, candidate, score):
    candidate_string = format_value(candidate)
    distance = hd(np.array(candidate), np.array(real)).sum()
    return f'{utils.color_fg.lightblue}{candidate_string} {utils.color_fg.green}Δ={distance:2d} {utils.color_fg.lightcyan}{score:5.1f}%  {utils.color_fg.orange}|{utils.color_fg.end}'


def get_edge_case_threshold(df, column):
    threshold_column = 'R' + column[1:]
    if threshold_column in df:
        threshold = df[(df.hw_v == 0) | (df.hw_v == 4)][threshold_column].mean()
    else:
        print('cannot find threshold column!')
        threshold_column = column
        threshold = None
    return threshold, threshold_column
