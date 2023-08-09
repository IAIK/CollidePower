# Collide+Power

The artifacts of the USENIX'23 security paper: [Collide+Power: Leaking Inaccessible Data with Software-based Power Side Channels](https://collidepower.com/paper/Collide+Power.pdf). The artifact appendix is appended to the paper. Additional information can be found [here](https://collidepower.com). These artifacts demonstrate Hamming distance leakage in shared CPU components (like the memory hierarchy) to extract inaccessible data over direct RAPL power readings or the Hertzbleed-based side channels. We **highlight** that we don't leak '*metadata*' like access patterns, but rather the actual **data**, e.g., the content of cache lines.

## Proof-of-Concept:
The PoCs are located in [poc](poc). Follow this [README.md](poc/README.md) for the cleaned-up poc. The PoC was designed for an 8-way L1 cache and 4-way L2 cache but due to the 16 used cache lines for eviction more ways could also work.

## Analysis:
The analysis framework is contained in [analyze](analyze). The folder contains the scripts to analyze and plot the recorded data. Follow this [README.md](analyze/README.md) for further details.

## Experiments:
The experiments using the poc and analysis framework are located in:

Variable                  | Description              | Minimal Runtime **ESTIMATES**
:-------------------------|:-------------------------|:-------------------------
[E1](E1_and_E2/README.md) | Basic leakage            | 10 hours
[E2](E1_and_E2/README.md) | Differential measurement | 0 (uses E1 data)
[E3](E3/README.md)        | CPA on the raw channel   | 5 hours
[E4](E4/README.md)        | Unmasked data influences | 5 hours
[E5](E5/README.md)        | MDS-Power                | 20 hours
[E6](E6/README.md)        | Meltdown-Power           | 50 hours

The minimal runtime is an estimate for a CPU with a good correlation. Your millage may vary. If the interface used does not work (E1) running the experiment for longer wont change the outcome. The log files should be **always valid** and can be copied to another machine without the need to stop the experiment. This allows for post-processing every few hours and verify if the experiment works. Letting the PoC run for longer will only increase the accuracy. If you don't want to stop the PoC, we usually copy away (scp) the CSV file during the experiment to another machine and post-process the CSV file there. 

To execute the experiments, please follow the preparation steps in the [poc/README.md](poc/README.md) and [analyze/README.md](analyze/README.md) readmes. After the test environment is configured all the experiments can be run after each other, no new configurations are required.

The run experiment run scripts will automatically set the correct [macros](poc/user/main.cpp#L42) for the PoC during building.
However, the overall system configuration is still mandatory (see [poc/README.md](poc/README.md)).

## Manual Configuration:
To use a manual configuration other than the one provided in the experiment folders follow the described in the [poc/README.md](poc/README.md).

# Warnings
**Warning #1**: We are providing this code as-is. You are responsible for protecting yourself, your property and data, and others from any risks caused by this code. This code may cause unexpected and undesirable behavior to occur on your machine.

**Warning #2**: This code is only for testing purposes. Do not run it on any production systems. Do not run it on any system that might be used by another person or entity.