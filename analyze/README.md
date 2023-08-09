# Analysis and Plotting Framework


# Requirements


Install `python3` and the requirements:

```
sudo apt install python3 python3-pip
```

And the required python3 packages with:

```
sudo pip3 install -r requirements.txt
```

# Configuration

We require energy and timing units of the system to calculate the exact leakage rates and scaling coefficients in Watt for the experiments.
Therefore, please execute [get_units.sh](../poc/get_units.sh) on the system where the [poc](../poc) is executed and enter the returned values in [unit_scaling.py](unit_scaling.py).

# Running the Analyze Script

## Correlate the Energy Model
Example of the initial correlation analysis from Table 2.

```bash
python3 cli.py \
  -i log.csv \
  --power.init confg 1 \
  -g Exp,v0,g0 \
  --per REnergyPP0,RTicks 5 95  \
  -u -g Exp  \
  --power.set_comp all \
  --power.find_coefs RPowerPP0 0
```

1) The command loads the CSV file.
2) The additional power-related columns are computed with scaling set to `config` (i.e, use the given config file).
3) We group the data based on the observable variables, i.e., the Experiments, the used guesses, and the unknown but constant victim value (see threat model).
4) We remove outliers that occur to scheduling etc.
5) We regroup only to consider the experiments.
6) We select all model components for the linear regression.
7) We compute the linear regression for the PP0 power measurements.

Example output:
```bash
> --power.find_coefs column='RPowerPP0', independent_coefficients=False
                                                rho  rho_l  rho_u     pv  hd_v0_g0  hd_v1_g1  hd_v1_v0  hd_g1_g0  hd_v1_g0     hw_g0     hw_g1     hw_v0     hw_v1       N       SNR
Exp                                                                                                                                                                                 
01_n00_n64_n01_PF_NOE_DPF_DSG_D_00         0.001825  -0.00   0.00   0.16  0.000000  0.000000  0.000195  0.000066  0.000098  0.000000  0.000063  0.000337  0.000118  601135  0.000000
02_n00_n64_n01_PF_L1E_DPF_DSG_D_00         0.088806   0.09   0.09   0.00  0.003258  0.003207  0.000000  0.000000  0.000000  0.010426  0.010537  0.001472  0.001315  575321  0.678784
03_n00_n64_n01_PF_L12EOpt2_DPF_DSG_D_00    0.144881   0.14   0.15   0.00  0.001531  0.001692  0.001866  0.021410  0.000654  0.006291  0.006394  0.000000  0.000000  578844  0.203599
04_n00_n64_n01_ST_NOE_DPF_DSG_D_00         0.003509   0.00   0.01   0.01  0.000041  0.000415  0.000211  0.000264  0.000077  0.000000  0.000000  0.000393  0.000084  582063  0.004702
05_n00_n64_n01_ST_L1E_DPF_DSG_D_00         0.062798   0.06   0.07   0.00  0.001529  0.001598  0.000000  0.003959  0.000000  0.004034  0.006128  0.000474  0.000430  572005  0.259520
06_n00_n64_n01_ST_L12EOpt2_DPF_DSG_D_00    0.251175   0.25   0.25   0.00  0.003916  0.005050  0.005152  0.061032  0.002637  0.010325  0.033747  0.000772  0.005074  598650  0.506089
07_n00_n64_n01_LD_NOE_DPF_DSG_D_00         0.074373   0.07   0.08   0.00  0.013612  0.000014  0.000000  0.000187  0.000028  0.008644  0.000081  0.000459  0.000345  569656  3.962873
08_n00_n64_n01_LD_L1E_DPF_DSG_D_00         0.610248   0.61   0.61   0.00  0.017648  0.008242  0.000000  0.000000  0.000000  0.177964  0.079366  0.010929  0.006136  581152  3.726701
09_n00_n64_n01_LD_L12EOpt2_DPF_DSG_D_00    0.499520   0.50   0.50   0.00  0.008374  0.003646  0.004012  0.042198  0.003011  0.096289  0.040578  0.007548  0.004854  587375  1.654727
10_n00_n64_n01_STNT_NOE_DPF_DSG_D_00       0.001390  -0.00   0.00   0.29  0.000000  0.000167  0.000000  0.000000  0.000165  0.000098  0.000094  0.000000  0.000000  583459  0.000729
11_n00_n64_n01_STNT_L1E_DPF_DSG_D_00       0.060876   0.06   0.06   0.00  0.001346  0.001516  0.000084  0.003504  0.000284  0.003924  0.005650  0.000787  0.000835  592903  0.234321
12_n00_n64_n01_STNT_L12EOpt2_DPF_DSG_D_00  0.243347   0.24   0.25   0.00  0.003951  0.005141  0.005546  0.059606  0.002653  0.010118  0.033187  0.001741  0.003478  608419  0.513221
```



# CPA
An example of the CPA similar to Figure 10.

```bash
python3 cli.py \
  -i log.csv \
  --power.init config 0 \
  --sel x.Id==7 \
  -g v0,g0 \
  --per REnergy,IEnergy 5 95 -u \
  --power.cpa DPower 100 3,5,10,20,30,50,100,200,300,500,750,1000,2000,3000,5000,10000 \
  --print --plot.line '1st' 1
```

1) Load the CSV.
2) Init power columns.
3) Select a specific experiment (with Id=7).
4) Group by observable experiment.
5) Remove outliers and ungroup.
6) Perform the CPA on the DPower column with 100 tests each using 3,5,... etc. samples.
7) Visualize the results for the percentage of the CPA with the correct value as the best candidate.


## All Commands

```
Usage: cli.py [OPTIONS] COMMAND1 [ARGS]... [COMMAND2 [ARGS]...]...

Options:
  -h, --help  Show this message and exit.

Commands:
  --csv                   Usage:  [OPTIONS]
                          > print csv of current data frame

  --cut                   Usage:  [OPTIONS] OUTLIERS LOWER UPPER
                          > cutaway outliers

  --filter                Usage:  [OPTIONS] COLUMNS [mean|median]
                          [NUMBER_SAMPLES]               > subtract moving
                          average filter

  --head                  Usage:  [OPTIONS]
                          > head of current data frame

  --idx                   Usage:  [OPTIONS] COLUMNS
                          > set data frame index

  --merge                 Usage:  [OPTIONS] ID_COLUMN IDS
                          > load intermediate data

  --mvper                 Usage:  [OPTIONS] OUTLIERS LOWER UPPER
                          [NUMBER_SAMPLES]                > cutaway outliers

  --mvstd                 Usage:  [OPTIONS] OUTLIERS LOWER UPPER
                          [NUMBER_SAMPLES]                > cutaway outliers

  --per                   Usage:  [OPTIONS] OUTLIERS LOWER UPPER
                          > remove percentile outliers

  --plot.filtered         Usage:  [OPTIONS] COLUMN INTERPOLATE
  --plot.heatmap          Usage:  [OPTIONS] COLUMN X Y
  --plot.hist             Usage:  [OPTIONS] COLUMN
  --plot.kde              Usage:  [OPTIONS] COLUMN
  --plot.line             Usage:  [OPTIONS] COLUMN XLOG
  --plot.line_xf          Usage:  [OPTIONS] COLUMN XLOG
  --plot.overview         Usage:  [OPTIONS] COLUMN
  --plot.scatter          Usage:  [OPTIONS] COLUMN
  --plot.show             Usage:  [OPTIONS]
  --plot.stats            Usage:  [OPTIONS] COLUMN
  --power.cpa             Usage:  [OPTIONS] COLUMN REPETITIONS N_SAMPLES
                          > perform a CPA on the data column

  --power.cpa_edge_cases  Usage:  [OPTIONS] DO_EDGE_CASES
                          > set of the CPA should also perform edge cases

  --power.find_coefs      Usage:  [OPTIONS] COLUMN INDEPENDENT_COEFFICIENTS
                          > find the model coefficients for the model
                          components

  --power.init            Usage:  [OPTIONS] [SCALE] EXPLODE
  --power.set_coefs       Usage:  [OPTIONS] COEFFICIENTS
                          > set the model coefficients

  --power.set_comp        Usage:  [OPTIONS] COMPONENTS
                          > set the model components

  --power.set_functions   Usage:  [OPTIONS] [pearson|spearman] [classic|huber]
                          > set the statistic functions to use

  --print                 Usage:  [OPTIONS]
                          > print current data frame

  --run                   Usage:  [OPTIONS] CODE
                          > run code snippet

  --sel                   Usage:  [OPTIONS] CODE
                          > select subset of data

  --std                   Usage:  [OPTIONS] OUTLIERS LOWER UPPER
                          > remove standard outliers

  -g                      Usage:  [OPTIONS] COLUMNS
                          > group data

  -i                      Usage:  [OPTIONS] FILE_NAME
                          > load csv file

  -l                      Usage:  [OPTIONS] ID
                          > load intermediate data

  -o                      Usage:  [OPTIONS] FILE_NAME
                          > store csv file

  -s                      Usage:  [OPTIONS] ID
                          > save intermediate data

  -u                      Usage:  [OPTIONS]
                          > ungroup data
```
