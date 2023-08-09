#include "cacheutils.h"
#include "config.h"
#include "interface.h"
#include "logging.h"
#include "memory.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdio.h>
#include <string.h>
#include <string_view>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

// string utils
#define xstr(s) str(s)
#define str(s)  #s

constexpr uint8_t PHYSICAL_CORES = 4;
constexpr uint8_t CORE           = 1;
constexpr uint8_t SIBLING        = CORE + 4; // required for -Power to work

// additional cores performing the Meltdown-power workload in background for amplification, set to zero if no additional cores should be used
constexpr uint8_t AMPLIFICATION_CORES = 0; // PHYSICAL_CORES - 1;

// choose the meltdown-power access type
constexpr int SYSCALL_TYPE = true ? DO_ACCESS : DO_RSB;

int fd;

#define PAPER_FIGURE_4
#define PAPER_TABLE_2_and_3
#define PAPER_FIGURE_9
#define PAPER_FIGURE_10
#define PAPER_FIGURE_12a_128x
#define PAPER_FIGURE_12a_1x
#define PAPER_FIGURE_13a
#define PAPER_FIGURE_15
/*
#undef PAPER_FIGURE_4
#undef PAPER_TABLE_2_and_3
#undef PAPER_FIGURE_9
#undef PAPER_FIGURE_10
#undef PAPER_FIGURE_12a_128x
#undef PAPER_FIGURE_12a_1x
#undef PAPER_FIGURE_13a
#undef PAPER_FIGURE_15
*/
// experiments contains the experiment for the config.
// each experiment is randomly sampled and executed

#if defined(PAPER_FIGURE_4)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 1, Inst::LOAD, Evict::NONE, 0, HWPF::DISABLED, CLFill::RANDOM, CLFill::RANDOM, 1 },
    { 2, 0, 128, 1, Inst::LOAD, Evict::L1, 0, HWPF::DISABLED, CLFill::RANDOM, CLFill::RANDOM, 1 },
    { 3, 0, 128, 1, Inst::LOAD, Evict::L1_L2, 0, HWPF::DISABLED, CLFill::RANDOM, CLFill::RANDOM, 1 },
};
#endif

#if defined(PAPER_TABLE_2_and_3)
static ExperimentConfig experiments[] = {
    // Prefetch
    { 1, 0, 64, 1, Inst::PREFETCH, Evict::NONE, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
    { 2, 0, 64, 1, Inst::PREFETCH, Evict::L1, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
    { 3, 0, 64, 1, Inst::PREFETCH, Evict::L1_L2, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
    // Store
    { 4, 0, 64, 1, Inst::STORE, Evict::NONE, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
    { 5, 0, 64, 1, Inst::STORE, Evict::L1, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
    { 6, 0, 64, 1, Inst::STORE, Evict::L1_L2, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
    // Load
    { 7, 0, 64, 1, Inst::LOAD, Evict::NONE, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
    { 8, 0, 64, 1, Inst::LOAD, Evict::L1, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
    { 9, 0, 64, 1, Inst::LOAD, Evict::L1_L2, 0, HWPF::DISABLED, CLFill::DISTINCT, CLFill::DISTINCT, 1 },
};
#endif

#if defined(PAPER_FIGURE_9)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 1, Inst::LOAD, Evict::L1, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 1 },
    { 2, 0, 128, 128, Inst::LOAD, Evict::L1, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 1 },
};
#endif

#if defined(PAPER_FIGURE_9_FAST)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 1, Inst::LOAD, Evict::L1, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 1 },
    //{ 2, 0, 128, 128, Inst::LOAD, Evict::L1, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 1 },
};
#endif

#if defined(PAPER_FIGURE_10)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 128, Inst::LOAD, Evict::NONE, 0, HWPF::DISABLED, CLFill::RANDOM, CLFill::ZEROS, 1 },
    { 2, 0, 128, 128, Inst::LOAD, Evict::NONE, 0, HWPF::DISABLED, CLFill::CONSTANT, CLFill::ZEROS, 1 },
    { 3, 0, 128, 128, Inst::LOAD, Evict::NONE, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 1 },
};
#endif

// FIGURE 11 reuses the data of PAPER_FIGURE_9

#if defined(PAPER_FIGURE_12a_128x)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 1, Inst::LOAD_MDS, Evict::NONE, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 2.095551 * 3 / 4 },
    { 2, 0, 128, 1, Inst::LOAD_MDS, Evict::L1, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 2.095551 * 3 / 4 },
    { 3, 0, 128, 1, Inst::LOAD_MDS, Evict::L1_L2, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 2.095551 * 3 / 4 },
};
#endif

#if defined(PAPER_FIGURE_12a_128x_FAST)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 1, Inst::LOAD_MDS, Evict::NONE, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 2.095551 * 3 / 4 },
    //{ 2, 0, 128, 1, Inst::LOAD_MDS, Evict::L1, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 2.095551 * 3 / 4 },
    //{ 3, 0, 128, 1, Inst::LOAD_MDS, Evict::L1_L2, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 2.095551 * 3 / 4 },
};
#endif

#if defined(PAPER_FIGURE_12a_1x)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 128, Inst::LOAD_MDS, Evict::NONE, 0, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 2.095551 * 3 / 4 },
};
#endif

#if defined(PAPER_FIGURE_13a)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 1, Inst::LOAD_MELTDOWN, Evict::L1_L2, 49, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 1.095551 * 0.1 },
    { 2, 0, 128, 1, Inst::STORE_MELTDOWN, Evict::L1_L2, 49, HWPF::DISABLED, CLFill::ZEROS, CLFill::ZEROS, 1.095551 * 0.1 },
};
#endif

#if defined(PAPER_FIGURE_15)
static ExperimentConfig experiments[] = {
    { 1, 0, 128, 1, Inst::LOAD, Evict::L1_L2, 0, HWPF::DISABLED, CLFill::DISTINCT_ADJACENT, CLFill::ZEROS, 1 },
    { 2, 0, 128, 1, Inst::LOAD, Evict::L1_L2, 0, HWPF::ENABLED, CLFill::DISTINCT_ADJACENT, CLFill::ZEROS, 1 },
};
#endif

// number of experiments
constexpr size_t N_experiments = sizeof(experiments) / sizeof(ExperimentConfig);

static test_data userspace_meltdown_experiment(ExperimentConfig const &config, uint8_t *target);
static test_data userspace_mds_experiment(ExperimentConfig const &config, uint8_t *target);
static test_data kernel_experiment(ExperimentConfig const &config, uint8_t *target);

// pin pthread to specific core
static void pin_core(uint8_t core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

// setup memory for experiment and dispatch
static test_data run(ExperimentConfig const &c, Memory const &m, CLFillValue v, CLFillValue g) {

    // get memory for experiment
    uint8_t *mem = m.get_memory(c.eviction, c.l1_set);

    // if we don't want any eviction we specify a different offset
    // this is only one part ... no eviction also requires specific instructions
    uint64_t offset = c.eviction == Evict::NONE ? 0x80 : 0;

    // we use an offset 0x80 instead of 0x40 since a page has 64 cache lines and we only use 17
    // therefore we still don't have any overlaps.

    for ( uint64_t page = 0; page < 17; ++page ) {
        // calculate cache line pointer
        uint8_t *cl = mem + page * (0x1000 + offset);

        // we choose the 2nd page as victim page
        bool is_victim = page == 1;

        // prepare the cache line
        if ( is_victim ) {
            v.fill_cl(c, cl);
        }
        else {
            g.fill_cl(c, cl);
        }
    }

    // configure the prefetchers
    set_pf_control(c.pf_control);

    // dispatch experiment
    switch ( c.instruction ) {
        case Inst::LOAD_MELTDOWN:
        case Inst::STORE_MELTDOWN:
            return userspace_meltdown_experiment(c, mem);

        case Inst::LOAD_MDS:
        case Inst::STORE_MDS:
            return userspace_mds_experiment(c, mem);

        default:
            return kernel_experiment(c, mem);
    };
}

static void do_access(uint8_t *target) {
    asm volatile(".align 64                       \n"
                 "1:                              \n"
                 "    mov %[nr], %%eax            \n"
                 "    syscall                     \n" // will perform movq   0x2000(%%rdx), %%xx
                 :
                 : [ nr ] "i"(__NR_ioctl), "D"(fd), "S"(DO_ACCESS), "d"(target)
                 : "rax", "rcx", "r11", "memory", "flags");
}

static uint8_t test[4096] = {};

[[gnu::noinline]] double test_flush_and_reload() {
    uint64_t count = 0;

    memset(test, 0xbb, 0x1000);

    for ( uint64_t i = 0; i < 10000; ++i ) {

        do_access((uint8_t *)test - 0x2000 + 48 * 64);

        count += flush_reload_t(test + 48 * 64);
    }

    return count / 10000.0;
}

int main() {

    // we open exactly our kernel module which is named after the parent's parent folder
    fd = open("/dev/" xstr(NAME), O_RDWR);
    if ( fd < 0 ) {
        printf("cannot open kernel module is it loaded? forgot sudo?\n");
        return -1;
    }

    double fr = test_flush_and_reload();

    fprintf(stderr, "# flush+reload: %f\n", fr);
    fprintf(stdout, "flush+reload: %f\n", fr);

    // load ptedit for memory finding
    if ( ptedit_init() ) {
        printf("Error: Could not initialize PTEditor, did you load the kernel module?\n");
        return -11;
    }

    // perform self inspection for debug purpose
    self_inspection();

    // we allocate 4 different test memories
    constexpr size_t MEMORIES = 4;

    std::vector<Memory> memories;

    // initialise the memories
    uint8_t *mem_base = (uint8_t *)0x7f5800000000;
    for ( size_t m = 0; m < MEMORIES; ++m ) {
        memories.emplace_back(mem_base);
    }

    // dump all possible information
    // print_system_config();

    // print csv header ... currently not nice
    fprintf(stderr, "TimeStamp,Value,Guess,ERep,SRep");
    ExperimentConfig::print_header();
    print_fields('R');
    print_fields('I');
    fprintf(stderr, "\n");

    // distributions for the experiments
    std::random_device rd;

    std::uniform_int_distribution<uint8_t> dist_experiment(0, N_experiments - 1);
    std::uniform_int_distribution<uint8_t> dist_memory(0, MEMORIES - 1);

    printf("CORE AND SIBLING %d %d <- please check!\n", CORE, SIBLING);

    for ( uint64_t index = 0; index < NUMBER_SAMPLES; ++index ) {

        // print current iteration
        fprintf(stdout, "\r%3zu/%3zu", index, NUMBER_SAMPLES);
        fflush(stdout);

        // select experiment and memory index
        uint8_t exp       = dist_experiment(rd);
        uint8_t mem_index = dist_memory(rd);

        ExperimentConfig const &c = experiments[exp];
        Memory const &          m = memories[mem_index];

        // repeat random selected variables for REP times
        for ( size_t experiment_rep = 0; experiment_rep < EXPERIMENT_REPEAT; ++experiment_rep ) {

            // sample cache line fills
            CLFillValue value = { c.v_fill, rd };
            CLFillValue guess = { c.g_fill, rd };

            // repeat random selected variables for REP times
            for ( size_t sample_rep = 0; sample_rep < SAMPLE_REPEAT; ++sample_rep ) {

                // perform differential measurement
                test_data i = run(c, m, value, guess.invert());
                test_data r = run(c, m, value, guess);

                // write the sample
                timestamp();
                fprintf(stderr, ",%2lu,%2lu,%2lu,%2lu", value.seed, guess.seed, experiment_rep, sample_rep);
                c.print_config(mem_index);
                print_data(r);
                print_data(i);
                fprintf(stderr, "\n");
            }
        }
    }

    printf("\n");

    return 0;
}

static void meltdown_caller(uint8_t *p, ExperimentConfig const &config, uint64_t count) {

    if ( config.instruction == Inst::STORE_MELTDOWN ) {
        asm volatile("lfence");
        asm volatile("    movq   0x1000(%%rdx), %%r8  \n" // save the guess values once
                     ".align 64                       \n"
                     "1:                              \n"

                     "    mov %[nr], %%eax            \n"
                     "    syscall                     \n" // will perform movq   0x2000(%%rdx), %%xx

                     "    movq %%r8,  0x1000(%%rdx)   \n"
                     "    movq %%r8,  0x3000(%%rdx)   \n"
                     "    movq %%r8,  0x4000(%%rdx)   \n"
                     "    movq %%r8,  0x5000(%%rdx)   \n"
                     "    movq %%r8,  0x9000(%%rdx)   \n"
                     "    movq %%r8,  0xa000(%%rdx)   \n"
                     "    movq %%r8,  0xb000(%%rdx)   \n"
                     "    movq %%r8,  0xc000(%%rdx)   \n"

                     "    mov %[nr], %%eax            \n"
                     "    syscall                     \n" // will perform movq   0x2000(%%rdx), %%xx

                     "    movq %%r8,  0x6000(%%rdx)   \n"
                     "    movq %%r8,  0x7000(%%rdx)   \n"
                     "    movq %%r8,  0x8000(%%rdx)   \n"
                     "    movq %%r8, 0x11000(%%rdx)   \n"
                     "    movq %%r8,  0xd000(%%rdx)   \n"
                     "    movq %%r8,  0xe000(%%rdx)   \n"
                     "    movq %%r8,  0xf000(%%rdx)   \n"
                     "    movq %%r8, 0x10000(%%rdx)   \n"

                     "    dec %[count]                \n"
                     "    jnz 1b                      \n"

                     :
                     : [ nr ] "i"(__NR_ioctl), "D"(fd), "S"(SYSCALL_TYPE), "d"(p), [ count ] "r"(count)
                     : "rax", "rcx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory", "flags");
    }
    else {

        if ( config.eviction == Evict::NONE ) {
            asm volatile("lfence");
            asm volatile(".align 64                       \n"
                         "1:                              \n"

                         "    mov %[nr], %%eax            \n"
                         "    syscall                     \n" // will perform movq   0x2000(%%rdx), %%xx

                         "    movq   0x1000(%%rdx), %%r8  \n"
                         "    movq   0x3100(%%rdx), %%r9  \n"
                         "    movq   0x4180(%%rdx), %%r10 \n"
                         "    movq   0x5200(%%rdx), %%r11 \n"
                         "    movq   0x9400(%%rdx), %%r12 \n"
                         "    movq   0xa480(%%rdx), %%r13 \n"
                         "    movq   0xb500(%%rdx), %%r14 \n"
                         "    movq   0xc580(%%rdx), %%r15 \n"

                         "    mov %[nr], %%eax            \n"
                         "    syscall                     \n" // will perform movq   0x2000(%%rdx), %%xx

                         "    movq   0x6280(%%rdx), %%r8  \n"
                         "    movq   0x7300(%%rdx), %%r9  \n"
                         "    movq   0x8380(%%rdx), %%r10 \n"
                         "    movq  0x11800(%%rdx), %%r11 \n"
                         "    movq   0xd600(%%rdx), %%r12 \n"
                         "    movq   0xe680(%%rdx), %%r13 \n"
                         "    movq   0xf700(%%rdx), %%r14 \n"
                         "    movq  0x10780(%%rdx), %%r15 \n"

                         "    dec %[count]                \n"
                         "    jnz 1b                      \n"

                         :
                         : [ nr ] "i"(__NR_ioctl), "D"(fd), "S"(SYSCALL_TYPE), "d"(p), [ count ] "r"(count)
                         : "rax", "rcx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory", "flags");
        }
        else {
            asm volatile("lfence");
            asm volatile(".align 64                       \n"
                         "1:                              \n"

                         "    mov %[nr], %%eax            \n"
                         "    syscall                     \n" // will perform movq   0x2000(%%rdx), %%xx

                         "    movq   0x1000(%%rdx), %%r8  \n"
                         "    movq   0x3000(%%rdx), %%r9  \n"
                         "    movq   0x4000(%%rdx), %%r10 \n"
                         "    movq   0x5000(%%rdx), %%r11 \n"
                         "    movq   0x9000(%%rdx), %%r12 \n"
                         "    movq   0xa000(%%rdx), %%r13 \n"
                         "    movq   0xb000(%%rdx), %%r14 \n"
                         "    movq   0xc000(%%rdx), %%r15 \n"

                         "    mov %[nr], %%eax            \n"
                         "    syscall                     \n" // will perform movq   0x2000(%%rdx), %%xx

                         "    movq   0x6000(%%rdx), %%r8  \n"
                         "    movq   0x7000(%%rdx), %%r9  \n"
                         "    movq   0x8000(%%rdx), %%r10 \n"
                         "    movq  0x11000(%%rdx), %%r11 \n"
                         "    movq   0xd000(%%rdx), %%r12 \n"
                         "    movq   0xe000(%%rdx), %%r13 \n"
                         "    movq   0xf000(%%rdx), %%r14 \n"
                         "    movq  0x10000(%%rdx), %%r15 \n"

                         "    dec %[count]                \n"
                         "    jnz 1b                      \n"

                         :
                         : [ nr ] "i"(__NR_ioctl), "D"(fd), "S"(SYSCALL_TYPE), "d"(p), [ count ] "r"(count)
                         : "rax", "rcx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory", "flags");
        }
    }
}

template<typename Prepare_t, typename Task_t>
static std::vector<int> fork_background_task(size_t n_cores, Prepare_t prepare, Task_t task) {

    // NOTE: this is not the best solution as std::atomic, could not work over shared memory, but for uint64_t it should be lockless anyway.
    static_assert(std::atomic<uint64_t>::is_always_lock_free);

    // allocate shared memory
    std::atomic<uint64_t> *mem = (std::atomic<uint64_t> *)mmap(nullptr, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE | MAP_ANONYMOUS, -1, 0);

    // init the sync variable
    std::atomic<uint64_t> *control = new (mem) std::atomic<uint64_t> { 0 };

    // pids
    std::vector<int> pids = {};
    pids.reserve(n_cores);

    // fork and invoke the function
    for ( uint8_t i = 0; i < n_cores; ++i ) {
        int pid = fork();
        if ( pid == 0 ) {

            prepare(i + 1);
            (*control)++;
            task(i + 1);

            exit(0);
        }
        else {
            pids.push_back(pid);
        }
    }

    while ( control->load() != (n_cores) ) {
        sched_yield();
    }

    // unmap shared control
    control->~atomic<uint64_t>();
    munmap(mem, 0x1000);

    return pids;
}

static void join_pids(std::vector<int> const &pids) {
    for ( auto pid : pids ) {
        kill(pid, SIGKILL);
        waitpid(pid, 0, 0);
    }
}

[[gnu::noinline, gnu::aligned(0x1000)]] static test_data userspace_meltdown_experiment(ExperimentConfig const &config, uint8_t *target) {

    if ( !(config.instruction == Inst::LOAD_MELTDOWN || config.instruction == Inst::STORE_MELTDOWN) ) {
        assert(false);
        printf("unkown case!\n");
        exit(-1);
    }

    if ( config.instruction == Inst::STORE_MELTDOWN && config.eviction == Evict::NONE ) {
        assert(false);
        printf("not supported!\n");
        exit(-1);
    }

    // we offset the target by one page so each load instruction is the same!
    uint8_t *p = target - 0x1000;

    // scale samples
    uint64_t count = LOOPS * config.sample_scale;

    // remove the victim page from the reach of the user
    mark_user_accessible(p + 0x2000, false);

    auto prepare = [&](uint8_t core_id) {
        // pin to core
        pin_core(core_id);
        // force tlb entry
        do_access(p);
        do_access(p);
    };

    auto background_task = [&](int core_id) {
        meltdown_caller(p, config, -1);
    };

    // if desired use multiple cores to amplify the leakage
    auto pids = fork_background_task(AMPLIFICATION_CORES, prepare, background_task);

    prepare(0);

    // sync energy measurement
    test_data start = measure();
    while ( measure().energy == start.energy )
        ;

    // start measuring
    start = measure();

    // invoke the victim loop
    meltdown_caller(p, config, count);

    // stop measuring
    test_data end = measure();

    // wait for the other processes to finish
    join_pids(pids);

    // bring the user mapping back
    mark_user_accessible(p + 0x2000, true);

    // store sample
    test_data data;
    memset(&data, 0, sizeof(test_data));
    data.energy     = end.energy - start.energy;
    data.energy_pp0 = end.energy_pp0 - start.energy_pp0;
    data.ticks      = end.ticks - start.ticks;
    return data;
}

[[gnu::noinline, gnu::aligned(0x1000)]] void mds_process(uint8_t *target) {

    asm volatile(".align 64                        \n"
                 "    nop                          \n"
                 ".align 64                        \n"
                 "    nop                          \n"
                 ".align 64                        \n"
                 "    nop                          \n"
                 ".align 64                        \n"
                 "    nop                          \n"
                 ".align 64                        \n"
                 "    nop                          \n"
                 ".align 64                        \n"
                 "    nop                          \n"
                 "1:                               \n"
                 "    movq 0x2000(%%rdx), %%r8     \n"
                 "    jmp 1b \n"
                 :
                 : "d"(target)
                 : "rax", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15");
}

[[gnu::noinline, gnu::aligned(0x1000)]] static test_data userspace_mds_experiment(ExperimentConfig const &config, uint8_t *target) {

    if ( !(config.instruction == Inst::LOAD_MDS || config.instruction == Inst::STORE_MDS) ) {
        assert(false);
        printf("unkown case!\n");
        exit(-1);
    }

    if ( config.instruction == Inst::STORE_MDS && config.eviction == Evict::NONE ) {
        assert(false);
        printf("not supported!\n");
        exit(-1);
    }

    bool do_stores = config.instruction == Inst::STORE_MDS;

    // we offset the target by one page so each load instruction is the same!
    uint8_t *p       = target - 0x1000;
    uint8_t *p_evict = p + (config.eviction == Evict::NONE ? 0x80 : 0);

    auto prepare = [](uint8_t core_id) {
        pin_core(SIBLING);
    };

    auto mds_thread = [=](uint8_t core_id) {
        mds_process(p_evict);
    };

    // create the victim process
    auto pids = fork_background_task(1, prepare, mds_thread);

    // remove the victim page from the attacker process (in that vadr mapping)
    munmap(p + 0x2000, 0x1000);
    // will be mapped back in memory.h in get_memory

    pin_core(CORE);

    // scale samples
    uint64_t count = LOOPS * config.sample_scale;

    /*{
        uint64_t *p = (uint64_t *)(target);
        printf(" [0x%llx] = 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx\n", (uint64_t)p, p[0], p[1], p[2],
               p[3], p[4], p[5], p[6], p[7]);
    }*/

    // sync energy measurement
    test_data start = measure();
    while ( measure().energy == start.energy )
        ;
    start = measure();

    if ( do_stores ) {
        asm volatile("lfence");
        asm volatile("    movq   0x1000(%%rdx), %%r8  \n"
                     ".align 64                       \n"
                     "   nop                          \n"
                     ".align 64                       \n"
                     "   nop                          \n"
                     ".align 64                       \n"
                     "   nop                          \n"
                     ".align 64                       \n"
                     "   nop                          \n"
                     ".align 64                       \n"
                     "   nop                          \n"
                     ".align 64                       \n"
                     "   nop                          \n"
                     ".align 64                       \n"
                     "1:                              \n"

                     "    movq %%r8,  0x1000(%%rdx)   \n"
                     "    nop                         \n"
                     "    movq %%r8,  0x3000(%%rdx)   \n"
                     "    movq %%r8,  0x4000(%%rdx)   \n"
                     "    movq %%r8,  0x5000(%%rdx)   \n"
                     "    movq %%r8,  0x6000(%%rdx)   \n"
                     "    movq %%r8,  0x7000(%%rdx)   \n"
                     "    movq %%r8,  0x8000(%%rdx)   \n"
                     "    movq %%r8,  0x9000(%%rdx)   \n"
                     "    movq %%r8,  0xa000(%%rdx)   \n"
                     "    movq %%r8,  0xb000(%%rdx)   \n"
                     "    movq %%r8,  0xc000(%%rdx)   \n"
                     "    movq %%r8,  0xd000(%%rdx)   \n"
                     "    movq %%r8,  0xe000(%%rdx)   \n"
                     "    movq %%r8,  0xf000(%%rdx)   \n"
                     "    movq %%r8, 0x10000(%%rdx)   \n"

                     "    dec %[count]                \n"
                     "    jnz 1b                      \n"

                     :
                     : "d"(p), [ count ] "r"(count)
                     : "rax", "rcx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory");
    }
    else {

        if ( config.eviction == Evict::NONE ) {
            asm volatile("lfence");
            asm volatile(".align 64                       \n"
                         "1:                              \n"

                         "    movq  0x1000(%%rdx), %%r8  \n"
                         "    nop                        \n"
                         "    movq  0x3100(%%rdx), %%r10 \n"
                         "    movq  0x4180(%%rdx), %%r11 \n"
                         "    movq  0x5200(%%rdx), %%r12 \n"
                         "    movq  0x6280(%%rdx), %%r13 \n"
                         "    movq  0x7300(%%rdx), %%r14 \n"
                         "    movq  0x8380(%%rdx), %%r15 \n"
                         "    movq  0x9400(%%rdx), %%r8  \n"
                         "    movq  0xa480(%%rdx), %%r9  \n"
                         "    movq  0xb500(%%rdx), %%r10 \n"
                         "    movq  0xc580(%%rdx), %%r11 \n"
                         "    movq  0xd600(%%rdx), %%r12 \n"
                         "    movq  0xe680(%%rdx), %%r13 \n"
                         "    movq  0xf700(%%rdx), %%r14 \n"
                         "    movq 0x10780(%%rdx), %%r15 \n"

                         "    dec %[count]                \n"
                         "    jnz 1b                      \n"

                         :
                         : "d"(p), [ count ] "r"(count)
                         : "rax", "rcx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15");
        }
        else {
            asm volatile("lfence");
            asm volatile(".align 64                       \n"
                         "1:                              \n"

                         "    movq   0x1000(%%rdx), %%r8  \n"
                         "    nop                         \n"
                         "    movq   0x3000(%%rdx), %%r10 \n"
                         "    movq   0x4000(%%rdx), %%r11 \n"
                         "    movq   0x5000(%%rdx), %%r12 \n"
                         "    movq   0x6000(%%rdx), %%r13 \n"
                         "    movq   0x7000(%%rdx), %%r14 \n"
                         "    movq   0x8000(%%rdx), %%r15 \n"
                         "    movq   0x9000(%%rdx), %%r8  \n"
                         "    movq   0xa000(%%rdx), %%r9  \n"
                         "    movq   0xb000(%%rdx), %%r10 \n"
                         "    movq   0xc000(%%rdx), %%r11 \n"
                         "    movq   0xd000(%%rdx), %%r12 \n"
                         "    movq   0xe000(%%rdx), %%r13 \n"
                         "    movq   0xf000(%%rdx), %%r14 \n"
                         "    movq  0x10000(%%rdx), %%r15 \n"

                         "    dec %[count]                \n"
                         "    jnz 1b                      \n"

                         :
                         : "d"(p), [ count ] "r"(count)
                         : "rax", "rcx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15");
        }
    }

    test_data end = measure();

    join_pids(pids);

    test_data data;
    memset(&data, 0, sizeof(test_data));
    data.energy     = end.energy - start.energy;
    data.energy_pp0 = end.energy_pp0 - start.energy_pp0;
    data.ticks      = end.ticks - start.ticks;

    return data;
}

[[gnu::noinline]] static test_data kernel_experiment(ExperimentConfig const &config, uint8_t *target) {
    unsigned long cmd = 0;

    pin_core(CORE);

    switch ( config.instruction ) {
        case Inst::LOAD:
            cmd = config.eviction == Evict::NONE ? TEST_LOAD_NO_EVICT : TEST_LOAD_L1_EVICT;
            break;
        case Inst::PREFETCH:
            cmd = config.eviction == Evict::NONE ? TEST_PREFETCH_NO_EVICT : TEST_PREFETCH_L1_EVICT;
            break;
        case Inst::STORE:
            cmd = config.eviction == Evict::NONE ? TEST_STORE_NO_EVICT : TEST_STORE_L1_EVICT;
            break;

        default:
            printf("unkown case!\n");
            exit(-1);
    }

    test_data data;
    memset(&data, 0, sizeof(test_data));
    data.address = (__u64)target;
    data.count   = LOOPS * config.sample_scale;

    if ( ioctl(fd, cmd, &data) < 0 ) {
        printf("ioctl error\n");
        exit(-1);
    }

    return data;
}