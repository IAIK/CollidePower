# Experiment 1 and 2
This experiments demonstrates the general leakage exploited by Collide+Power.

## Preliminary
Follow the preliminary steps of this [README.md](../README.md).

## How to Run
Execute the following command and let the script run for the minimal estimated time.

```
./run_e1_e2.sh log.csv
```

## How to Analyze

Execute the following command to analyze the results of E1.
```
./pp_e1.sh log.csv
```

Execute the following command to analyze the results of E2.
```
./pp_e2.sh log.csv
```

## Example output
The example output for the `pp_e1.sh` script with a short test run.
To derive the exact coefficients for the Hamming distance and weight (hd_x_y and hw_z), the values need to be divided by 64 which is the amplification in this scenario.

```
> --power.find_coefs column='RPowerPP0', independent_coefficients=False	
                                                rho  rho_l  rho_u     pv  hd_v0_g0  hd_v1_g1  hd_v1_v0  hd_g1_g0  hd_v0_g1  hd_v1_g0     hw_g0     hw_g1     hw_v0     hw_v1     N       SNR
Exp                                                                                                                                                                                         
01_n00_n64_n01_PF_NOE_DPF_DSG_D_00         0.032144  -0.00   0.06   0.05  0.000000  0.003415  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.013242  3695  0.067258
02_n00_n64_n01_PF_L1E_DPF_DSG_D_00         0.116619   0.09   0.15   0.00  0.003815  0.000959  0.000000  0.000000  0.000000  0.000000  0.010032  0.009013  0.001270  0.000000  4113  1.117392
03_n00_n64_n01_PF_L12EOpt2_DPF_DSG_D_00    0.190344   0.16   0.22   0.00  0.008465  0.000000  0.000000  0.029911  0.000000  0.000000  0.017288  0.012286  0.001150  0.002171  3673  1.877840
04_n00_n64_n01_ST_NOE_DPF_DSG_D_00         0.049990   0.02   0.08   0.00  0.000000  0.000000  0.008478  0.002710  0.000000  0.005025  0.015144  0.000075  0.000000  0.000000  3204  0.000000
05_n00_n64_n01_ST_L1E_DPF_DSG_D_00         0.103825   0.07   0.14   0.00  0.004298  0.000000  0.000000  0.007696  0.000000  0.000000  0.007489  0.017363  0.000000  0.000062  3523  0.493694
06_n00_n64_n01_ST_L12EOpt2_DPF_DSG_D_00    0.252075   0.22   0.28   0.00  0.004303  0.005123  0.004293  0.056952  0.003202  0.005484  0.013322  0.029164  0.002665  0.001239  3904  0.646234
07_n00_n64_n01_LD_NOE_DPF_DSG_D_00         0.034151   0.00   0.06   0.03  0.002949  0.001853  0.006445  0.000130  0.000000  0.000000  0.006699  0.001743  0.000000  0.000000  4053  0.142257
08_n00_n64_n01_LD_L1E_DPF_DSG_D_00         0.669704   0.65   0.69   0.00  0.014462  0.001759  0.001420  0.000000  0.002856  0.000000  0.172728  0.091507  0.005546  0.004461  3855  2.505354
09_n00_n64_n01_LD_L12EOpt2_DPF_DSG_D_00    0.438033   0.41   0.46   0.00  0.009458  0.000753  0.012305  0.046379  0.004737  0.000023  0.116806  0.064775  0.011271  0.000000  4226  0.903964
10_n00_n64_n01_STNT_NOE_DPF_DSG_D_00       0.035346   0.00   0.07   0.03  0.001904  0.005660  0.006952  0.004810  0.001819  0.000000  0.000000  0.000000  0.000000  0.000000  3571  0.420805
11_n00_n64_n01_STNT_L1E_DPF_DSG_D_00       0.068288   0.04   0.10   0.00  0.000218  0.000000  0.001754  0.006287  0.000000  0.000089  0.004834  0.010688  0.000000  0.000000  4519  0.001243
12_n00_n64_n01_STNT_L12EOpt2_DPF_DSG_D_00  0.236996   0.20   0.27   0.00  0.001790  0.007921  0.012318  0.045537  0.008656  0.006720  0.016837  0.038320  0.004332  0.010107  2607  0.881185
```