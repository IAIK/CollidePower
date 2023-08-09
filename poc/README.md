# Collide+Power - PoC

# Requirements

## Ubuntu Packages

To build the PoC, we need to install a compiler and additional tools.

```
sudo apt install cmake build-essential git python3 python3-pip g++ gcc python3 cpuid
sudo apt install linux-tools-$(uname -r)
sudo apt install linux-tools-$(uname -r)-generic
sudo apt install linux-headers-$(uname -r)
```

## PTEditor
The PoC requires [PTEditor](https://github.com/misc0110/PTEditor/tree/04e6acf54c5fc9cdd025331454ad45aa9e04119a) (commit: 4e6acf). Please follow the readmes there to load the kernel module.

## Grub
To achieve a suitable environment for the leakage analysis, we run with the grub command line:

```
quiet splash mitigations=off nmi_watchdog=0 isolcpus=domain,managed_irq,1,2,3,4,5,7,8,9,10,11 nohz_full=1,2,3,4,5,7,8,9,10,11 nosmep nosmap intel_pstate=per_cpu_perf_limits dis_ucode_ldr
```

Where the CPU list needs to be adopted based on the CPU's number of hardware threads, e.g., for six physical cores, we start with '1' and go until '11'. Thread '0' will not be isolated. Check `lscpu -e` to see the number of threads.

To set the grub command line follow these steps:

```
sudo vi /etc/default/grub
# add the parameters to GRUB_CMDLINE_LINUX_DEFAULT and save the file
sudo update-grub
# reboot the machine
```

## Huge Pages
We use huge pages to build the eviction sets for L1 and L2. Please use the following command to enable transparent huge pages:

```
sudo sysctl -w vm.nr_hugepages=32
```

Validate that huge pages were allocated with:

```
cat /proc/meminfo | grep "HugePages_" 
```

Please ensure that:

```
cat /sys/kernel/mm/transparent_hugepage/enabled
```

Returns:

```
always [madvise] never
```

Otherwise run:

```
echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
```

## Configuration

Define the following variables in [main.cpp](user/main.cpp):

```c++
constexpr uint8_t PHYSICAL_CORES = 4;
constexpr uint8_t CORE           = 1;
constexpr uint8_t SIBLING        = CORE + 4; // required for MDS-Power to work

// additional cores performing the Meltdown-power workload in background for amplification, set to zero if no additional cores should be used
constexpr uint8_t AMPLIFICATION_CORES = 0; // PHYSICAL_CORES - 1;

// choose the meltdown-power access type
constexpr int SYSCALL_TYPE = true ? DO_ACCESS : DO_RSB;
```

Variable | Description
:-------------------------|:-------------------------
PHYSICAL_CORES | number of physical cores of the CPU
CORE | the core where the PoC should be pinned
SIBLING | the hyper/sibling-thread to the CORE (required for MDS-Power)
AMPLIFICATION_CORES | meltdown-power can be spawned on multiple cores if desired
SYSCALL_TYPE | define if meltdown-power should use direct access in the kernel or the artificial RSB gadget

Use `lscpu -e` if unsure.

## Note: Duration of the Samples
The duration of the inner loop can be adapted by the last parameter of an `experiment`:

```c++
{ ... , 2.095551 * 3 / 4 },
```

This changes how fast a sample is recorded and influences the outcome of the CPA, as the inner loop is the primary amplification for energy consumption. We selected these values based on previous results. Furthermore, if this parameters is set to low, the RAPL interface might not observe any updates and the *consumed* energy will be reported as zero.


# Plot

For plotting the data, please refer to [here](../analyze/README.md).

# Manual Experiment Selection

The current source is designed to be invoked with the given run scripts. If the source should be compiled without such a script, one of the experiments must be enabled. Only one experiment should and can be enabled at a given time!

```c++
#define PAPER_FIGURE_4
#define PAPER_TABLE_2_and_3
#define PAPER_FIGURE_9
#define PAPER_FIGURE_10
#define PAPER_FIGURE_12a_128x
#define PAPER_FIGURE_12a_1x
#define PAPER_FIGURE_13a
#define PAPER_FIGURE_15
```

This will set the corresponding experiments in the `experiments` array.

To run the manual selected experiment use:

Intel:
```
cd user
make
make -C ../module clean all load
sudo ./main 2> log.csv
```

AMD:
```
cd user
make
make -C ../module clean amd load
sudo ./main 2> log.csv
```