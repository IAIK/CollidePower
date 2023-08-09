#include "interface.h"

#include <linux/kallsyms.h>
#include <linux/kernel.h>
#include <linux/kprobes.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <linux/signal.h>
#include <linux/types.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include <linux/wait.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Andreas Kogler");
MODULE_DESCRIPTION("");
MODULE_VERSION("");

#undef MSR_RAPL_POWER_UNIT
#undef MSR_PKG_ENERGY_STATUS
#undef MSR_DRAM_ENERGY_STATUS
#undef MSR_PP0_ENERGY_STATUS
#undef MSR_IA32_MISC_ENABLE
#undef MSR_MISC_FEATURE_CONTROL
#undef MSR_APERF
#undef MSR_MPERF

#if !defined(IS_AMD)
#pragma message "### Building INTEL"
#define MSR_RAPL_POWER_UNIT      0x00000606
#define MSR_PKG_ENERGY_STATUS    0x00000611
#define MSR_DRAM_ENERGY_STATUS   0x00000619
#define MSR_PP0_ENERGY_STATUS    0x00000639
#define MSR_IA32_MISC_ENABLE     0x1A0
#define MSR_MISC_FEATURE_CONTROL 0x1A4
#define MSR_APERF                0xE8
#define MSR_MPERF                0xE7
#else
#pragma message "### Building AMD"
#define MSR_RAPL_POWER_UNIT   0xC0010299
#define MSR_PP0_ENERGY_STATUS 0xC001029A
#define MSR_PKG_ENERGY_STATUS 0xC001029B
#endif

#define ENERGY() RDMSR(MSR_PKG_ENERGY_STATUS)

/********************************************************************************
 * logging
 ********************************************************************************/
#define VA_ARGS(...) , ##__VA_ARGS__

#define xstr(s) str(s)
#define str(s)  #s

#define TAG xstr(NAME)

#define INFO(_x, ...)  printk(KERN_INFO "[" TAG "] " _x "\n" VA_ARGS(__VA_ARGS__))
#define ERROR(_x, ...) printk(KERN_ERR "[" TAG "] " _x "\n" VA_ARGS(__VA_ARGS__))

/********************************************************************************
 * device functions
 ********************************************************************************/
static long module_ioctl(struct file *filp, unsigned int cmd, unsigned long arg);
static int  module_open(struct inode *, struct file *);
static int  module_release(struct inode *, struct file *);

typedef unsigned long (*kallsyms_lookup_name_t)(const char *);
typedef unsigned long (*change_protection_t)(struct vm_area_struct *vma, unsigned long start, unsigned long end, pgprot_t newprot, unsigned long cp_flags);

change_protection_t ptr_change_protection;

static struct file_operations file_ops = { .open = module_open, .release = module_release, .unlocked_ioctl = module_ioctl };

static struct miscdevice dev = { .minor = MISC_DYNAMIC_MINOR, .name = TAG, .fops = &file_ops, .mode = S_IRUGO | S_IWUGO };

static uint32_t device_open_count = 0;

static uint64_t direct_physical_map_address = 0;

/********************************************************************************
 * access macros
 ********************************************************************************/

#define WRMSR(_nr, _val) wrmsrl(_nr, _val)
#define RDMSR(_nr)       __rdmsr(_nr)

//#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 7, 0)
unsigned long local_kallsyms_lookup_name(const char *name) {
    kallsyms_lookup_name_t kallsyms_lookup_name;

    struct kprobe kp = {
        .symbol_name = "kallsyms_lookup_name",
    };

    int ret = register_kprobe(&kp);
    if ( ret < 0 ) {
        return 0;
    }

    unregister_kprobe(&kp);

    if ( kp.addr == 0 ) {
        return 0;
    }

    kallsyms_lookup_name = (kallsyms_lookup_name_t)kp.addr;

    return kallsyms_lookup_name(name);
}
//#endif

/********************************************************************************
 * IOCTL
 ********************************************************************************/

uint64_t get_direct_physical_map_address(void) {
    unsigned long addr = local_kallsyms_lookup_name("page_offset_base");
    if ( addr == 0 ) {
        ERROR("could not recover direct physical map address!");
        return 0;
    }
    return *(uint64_t *)addr;
}

change_protection_t get_change_protection_function(void) {
    unsigned long addr = local_kallsyms_lookup_name("change_protection");
    if ( addr == 0 ) {
        ERROR("cannot get change protection function!");
        return 0;
    }
    return (change_protection_t)addr;
}

#if !defined(IS_AMD)
#define TEST_FUNCTION(_name, _cl_offset, _instructions)                                                                                                                          \
    long __attribute__((aligned(0x1000))) _name(struct file *filep, unsigned int cmd, unsigned long arg) {                                                                       \
                                                                                                                                                                                 \
        struct test_data *data = (struct test_data *)arg;                                                                                                                        \
                                                                                                                                                                                 \
        uint64_t mem = data->address - 0x1000;                                                                                                                                   \
                                                                                                                                                                                 \
        uint32_t start = 0;                                                                                                                                                      \
                                                                                                                                                                                 \
        uint32_t energy_start     = 0;                                                                                                                                           \
        uint32_t energy_end       = 0;                                                                                                                                           \
        uint32_t energy_start_pp0 = 0;                                                                                                                                           \
        uint32_t energy_end_pp0   = 0;                                                                                                                                           \
        uint64_t ticks_start      = 0;                                                                                                                                           \
        uint64_t ticks_end        = 0;                                                                                                                                           \
        uint64_t perf_status      = 0;                                                                                                                                           \
        uint64_t therm_status     = 0;                                                                                                                                           \
        uint64_t therm_target     = 0;                                                                                                                                           \
        uint64_t mperf_start      = 0;                                                                                                                                           \
        uint64_t mperf_end        = 0;                                                                                                                                           \
        uint64_t aperf_start      = 0;                                                                                                                                           \
        uint64_t aperf_end        = 0;                                                                                                                                           \
                                                                                                                                                                                 \
        uint64_t dro = 0;                                                                                                                                                        \
        uint64_t tcc = 0;                                                                                                                                                        \
                                                                                                                                                                                 \
        /*int       i;                                                                                                                                                           \
        uint64_t *p;                                                                                                                                                             \
        printk("%s: \n", #_name);                                                                                                                                                \
        for ( i = 0; i < 16; ++i ) {                                                                                                                                             \
            p = (uint64_t *)(mem + i * (0x1000 + _cl_offset) + 0x1000);                                                                                                          \
            printk("%x [0x%llx] = 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx\n", i, (uint64_t)p, p[0],                                                              \
                   p[1], p[2], p[3], p[4], p[5], p[6], p[7]);                                                                                                                    \
        }*/                                                                                                                                                                      \
                                                                                                                                                                                 \
        start = ENERGY();                                                                                                                                                        \
                                                                                                                                                                                 \
        while ( start == ENERGY() )                                                                                                                                              \
            ;                                                                                                                                                                    \
                                                                                                                                                                                 \
        asm volatile(                                                                                                                                                            \
                                                                                                                                                                                 \
            "    cli                           \n"                                                                                                                               \
                                                                                                                                                                                 \
            /* mperf aperf start*/                                                                                                                                               \
            "    mov $0xE8, %%ecx              \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[aperf_start]     \n"                                                                                                                               \
            "    mov %%edx, -4+%H[aperf_start] \n"                                                                                                                               \
                                                                                                                                                                                 \
            "    mov $0xE7, %%ecx              \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[mperf_start]     \n"                                                                                                                               \
            "    mov %%edx, -4+%H[mperf_start] \n"                                                                                                                               \
                                                                                                                                                                                 \
            /* tsc start*/                                                                                                                                                       \
            "    xor %%eax, %%eax              \n"                                                                                                                               \
            "    cpuid                         \n"                                                                                                                               \
            "    rdtsc                         \n"                                                                                                                               \
            "    mov %%eax, %[ticks_start]     \n"                                                                                                                               \
            "    mov %%edx, -4+%H[ticks_start] \n"                                                                                                                               \
                                                                                                                                                                                 \
            /* energy start*/                                                                                                                                                    \
            "    mov $0x611, %%ecx             \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[energy_start]    \n" /*   skip edx*/                                                                                                               \
                                                                                                                                                                                 \
            /* energy start*/                                                                                                                                                    \
            "    mov $0x639, %%ecx             \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[energy_start_pp0]   \n" /*   skip edx*/                                                                                                            \
                                                                                                                                                                                 \
            ".align 64                         \n"                                                                                                                               \
            "    nop                           \n"                                                                                                                               \
            ".align 64                         \n"                                                                                                                               \
            "1:                                \n" _instructions                                                                                                                 \
                                                                                                                                                                                 \
            "    dec %[count]                  \n"                                                                                                                               \
            "    jnz 1b                        \n"                                                                                                                               \
                                                                                                                                                                                 \
            /* energy end*/                                                                                                                                                      \
            "    mov  $0x639, %%ecx            \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[energy_end_pp0]  \n" /*   skip edx*/                                                                                                               \
                                                                                                                                                                                 \
            /* energy end*/                                                                                                                                                      \
            "    mov  $0x611, %%ecx            \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[energy_end]      \n" /*   skip edx*/                                                                                                               \
                                                                                                                                                                                 \
            /* read pstate and voltage*/                                                                                                                                         \
            "    mov $0x198, %%ecx             \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[perf_status]     \n"                                                                                                                               \
            "    mov %%edx, -4+%H[perf_status] \n"                                                                                                                               \
                                                                                                                                                                                 \
            /* mperf aperf end*/                                                                                                                                                 \
            "    mov $0xE8, %%ecx              \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[aperf_end]       \n"                                                                                                                               \
            "    mov %%edx, -4+%H[aperf_end]   \n"                                                                                                                               \
                                                                                                                                                                                 \
            "    mov $0xE7, %%ecx              \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[mperf_end]       \n"                                                                                                                               \
            "    mov %%edx, -4+%H[mperf_end]   \n"                                                                                                                               \
                                                                                                                                                                                 \
            /* tsc end*/                                                                                                                                                         \
            "    rdtscp                        \n"                                                                                                                               \
            "    mov %%eax, %[ticks_end]       \n"                                                                                                                               \
            "    mov %%edx, -4+%H[ticks_end]   \n"                                                                                                                               \
            "    xor %%eax, %%eax              \n"                                                                                                                               \
            "    cpuid                         \n"                                                                                                                               \
                                                                                                                                                                                 \
            /* read thermal status*/                                                                                                                                             \
            "    mov $0x19c, %%ecx             \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[therm_status]    \n" /*   skip edx*/                                                                                                               \
                                                                                                                                                                                 \
            /* read temperature target*/                                                                                                                                         \
            "    mov $0x1a2, %%ecx             \n"                                                                                                                               \
            "    rdmsr                         \n"                                                                                                                               \
            "    mov %%eax, %[therm_target]    \n" /*   skip edx*/                                                                                                               \
                                                                                                                                                                                 \
            "    sti                           \n"                                                                                                                               \
                                                                                                                                                                                 \
            : [ energy_start ] "=m"(energy_start), [ energy_end ] "=m"(energy_end), [ ticks_start ] "=m"(ticks_start), [ mperf_start ] "=m"(mperf_start),                        \
              [ aperf_start ] "=m"(aperf_start), [ mperf_end ] "=m"(mperf_end), [ aperf_end ] "=m"(aperf_end), [ ticks_end ] "=m"(ticks_end), [ perf_status ] "=m"(perf_status), \
              [ therm_status ] "=m"(therm_status), [ therm_target ] "=m"(therm_target), [ energy_start_pp0 ] "=m"(energy_start_pp0), [ energy_end_pp0 ] "=m"(energy_end_pp0)     \
            : [ count ] "r"(data->count), [ mem ] "r"(mem)                                                                                                                       \
            : "rax", "rbx", "rcx", "rdx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory", "flags");                                                              \
                                                                                                                                                                                 \
        dro = (therm_status >> 16) & 0x7F;                                                                                                                                       \
        tcc = (therm_target >> 16) & 0xFF;                                                                                                                                       \
                                                                                                                                                                                 \
        data->energy      = energy_end - energy_start;                                                                                                                           \
        data->energy_pp0  = energy_end_pp0 - energy_start_pp0;                                                                                                                   \
        data->ticks       = ticks_end - ticks_start;                                                                                                                             \
        data->pstate      = ((perf_status & 0xFFFFllu) >> 8);                                                                                                                    \
        data->voltage     = ((perf_status & 0xFFFF00000000llu) >> 32);                                                                                                           \
        data->temperature = tcc - dro;                                                                                                                                           \
        data->mperf       = mperf_end - mperf_start;                                                                                                                             \
        data->aperf       = aperf_end - aperf_start;                                                                                                                             \
                                                                                                                                                                                 \
        return 0;                                                                                                                                                                \
    }
#else // AMD
#define TEST_FUNCTION(_name, _cl_offset, _instructions)                                                                                                                          \
    long __attribute__((aligned(0x1000))) _name(struct file *filep, unsigned int cmd, unsigned long arg) {                                                                       \
                                                                                                                                                                                 \
        struct test_data *data = (struct test_data *)arg;                                                                                                                        \
                                                                                                                                                                                 \
        uint64_t mem = data->address - 0x1000;                                                                                                                                   \
                                                                                                                                                                                 \
        uint32_t start = 0;                                                                                                                                                      \
                                                                                                                                                                                 \
        uint32_t energy_start     = 0;                                                                                                                                           \
        uint32_t energy_end       = 0;                                                                                                                                           \
        uint32_t energy_start_pp0 = 0;                                                                                                                                           \
        uint32_t energy_end_pp0   = 0;                                                                                                                                           \
        uint64_t ticks_start      = 0;                                                                                                                                           \
        uint64_t ticks_end        = 0;                                                                                                                                           \
        uint64_t perf_status      = 0;                                                                                                                                           \
        uint64_t therm_status     = 0;                                                                                                                                           \
        uint64_t therm_target     = 0;                                                                                                                                           \
        uint64_t mperf_start      = 0;                                                                                                                                           \
        uint64_t mperf_end        = 0;                                                                                                                                           \
        uint64_t aperf_start      = 0;                                                                                                                                           \
        uint64_t aperf_end        = 0;                                                                                                                                           \
                                                                                                                                                                                 \
        uint64_t dro = 0;                                                                                                                                                        \
        uint64_t tcc = 0;                                                                                                                                                        \
                                                                                                                                                                                 \
        /*int       i;                                                                                                                                                           \
        uint64_t *p;                                                                                                                                                             \
        printk("%s: \n", #_name);                                                                                                                                                \
        for ( i = 0; i < 16; ++i ) {                                                                                                                                             \
            p = (uint64_t *)(mem + i * (0x1000 + _cl_offset) + 0x1000);                                                                                                          \
            printk("%x [0x%llx] = 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx 0x%llx\n", i, (uint64_t)p, p[0],                                                              \
                   p[1], p[2], p[3], p[4], p[5], p[6], p[7]);                                                                                                                    \
        }*/                                                                                                                                                                      \
                                                                                                                                                                                 \
        start = ENERGY();                                                                                                                                                        \
                                                                                                                                                                                 \
        while ( start == ENERGY() )                                                                                                                                              \
            ;                                                                                                                                                                    \
                                                                                                                                                                                 \
        asm volatile(                                                                                                                                                            \
                                                                                                                                                                                 \
            "    cli                            \n"                                                                                                                              \
                                                                                                                                                                                 \
            /* tsc start*/                                                                                                                                                       \
            "    xor %%eax, %%eax               \n"                                                                                                                              \
            "    cpuid                          \n"                                                                                                                              \
            "    rdtsc                          \n"                                                                                                                              \
            "    mov %%eax, %[ticks_start]      \n"                                                                                                                              \
            "    mov %%edx, -4+%H[ticks_start]  \n"                                                                                                                              \
                                                                                                                                                                                 \
            /* energy start*/                                                                                                                                                    \
            "    mov $0xC001029B, %%rcx         \n"                                                                                                                              \
            "    rdmsr                          \n"                                                                                                                              \
            "    mov %%eax, %[energy_start]     \n" /*   skip edx*/                                                                                                              \
                                                                                                                                                                                 \
            /* energy start*/                                                                                                                                                    \
            "    mov $0xC001029A, %%rcx         \n"                                                                                                                              \
            "    rdmsr                          \n"                                                                                                                              \
            "    mov %%eax, %[energy_start_pp0] \n" /*   skip edx*/                                                                                                              \
                                                                                                                                                                                 \
            ".align 64                          \n"                                                                                                                              \
            "    nop                            \n"                                                                                                                              \
            ".align 64                          \n"                                                                                                                              \
            "1:                                 \n" _instructions                                                                                                                \
                                                                                                                                                                                 \
            "    dec %[count]                   \n"                                                                                                                              \
            "    jnz 1b                         \n"                                                                                                                              \
                                                                                                                                                                                 \
            /* energy end*/                                                                                                                                                      \
            "    mov  $0xC001029A, %%rcx        \n"                                                                                                                              \
            "    rdmsr                          \n"                                                                                                                              \
            "    mov %%eax, %[energy_end_pp0]   \n" /*   skip edx*/                                                                                                              \
                                                                                                                                                                                 \
            /* energy end*/                                                                                                                                                      \
            "    mov  $0xC001029B, %%rcx        \n"                                                                                                                              \
            "    rdmsr                          \n"                                                                                                                              \
            "    mov %%eax, %[energy_end]       \n" /*   skip edx*/                                                                                                              \
                                                                                                                                                                                 \
            /* tsc end*/                                                                                                                                                         \
            "    rdtscp                         \n"                                                                                                                              \
            "    mov %%eax, %[ticks_end]        \n"                                                                                                                              \
            "    mov %%edx, -4+%H[ticks_end]    \n"                                                                                                                              \
            "    xor %%eax, %%eax               \n"                                                                                                                              \
            "    cpuid                          \n"                                                                                                                              \
                                                                                                                                                                                 \
            "    sti                            \n"                                                                                                                              \
                                                                                                                                                                                 \
            : [ energy_start ] "=m"(energy_start), [ energy_end ] "=m"(energy_end), [ ticks_start ] "=m"(ticks_start), [ mperf_start ] "=m"(mperf_start),                        \
              [ aperf_start ] "=m"(aperf_start), [ mperf_end ] "=m"(mperf_end), [ aperf_end ] "=m"(aperf_end), [ ticks_end ] "=m"(ticks_end), [ perf_status ] "=m"(perf_status), \
              [ therm_status ] "=m"(therm_status), [ therm_target ] "=m"(therm_target), [ energy_start_pp0 ] "=m"(energy_start_pp0), [ energy_end_pp0 ] "=m"(energy_end_pp0)     \
            : [ count ] "r"(data->count), [ mem ] "r"(mem)                                                                                                                       \
            : "rax", "rbx", "rcx", "rdx", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "memory", "flags");                                                              \
                                                                                                                                                                                 \
        dro = (therm_status >> 16) & 0x7F;                                                                                                                                       \
        tcc = (therm_target >> 16) & 0xFF;                                                                                                                                       \
                                                                                                                                                                                 \
        data->energy      = energy_end - energy_start;                                                                                                                           \
        data->energy_pp0  = energy_end_pp0 - energy_start_pp0;                                                                                                                   \
        data->ticks       = ticks_end - ticks_start;                                                                                                                             \
        data->pstate      = ((perf_status & 0xFFFFllu) >> 8);                                                                                                                    \
        data->voltage     = ((perf_status & 0xFFFF00000000llu) >> 32);                                                                                                           \
        data->temperature = tcc - dro;                                                                                                                                           \
        data->mperf       = mperf_end - mperf_start;                                                                                                                             \
        data->aperf       = aperf_end - aperf_start;                                                                                                                             \
                                                                                                                                                                                 \
        return 0;                                                                                                                                                                \
    }

#endif

TEST_FUNCTION(test_load_no_evict, 0x80,
              "    mov   0x1000(%[mem]), %%r8  \n"
              "    mov   0x2080(%[mem]), %%r9  \n"
              "    mov   0x3100(%[mem]), %%r10 \n"
              "    mov   0x4180(%[mem]), %%r11 \n"
              "    mov   0x5200(%[mem]), %%r12 \n"
              "    mov   0x6280(%[mem]), %%r13 \n"
              "    mov   0x7300(%[mem]), %%r14 \n"
              "    mov   0x8380(%[mem]), %%r15 \n"

              "    mov   0x9400(%[mem]), %%r8  \n"
              "    mov   0xa480(%[mem]), %%rcx \n"
              "    mov   0xb500(%[mem]), %%r10 \n"
              "    mov   0xc580(%[mem]), %%r11 \n"
              "    mov   0xd600(%[mem]), %%r12 \n"
              "    mov   0xe680(%[mem]), %%r13 \n"
              "    mov   0xf700(%[mem]), %%r14 \n"
              "    mov  0x10780(%[mem]), %%r15 \n")

TEST_FUNCTION(test_load_l1_evict, 0,
              "    mov   0x1000(%[mem]), %%r8  \n"
              "    mov   0x2000(%[mem]), %%r9  \n"
              "    mov   0x3000(%[mem]), %%r10 \n"
              "    mov   0x4000(%[mem]), %%r11 \n"
              "    mov   0x5000(%[mem]), %%r12 \n"
              "    mov   0x6000(%[mem]), %%r13 \n"
              "    mov   0x7000(%[mem]), %%r14 \n"
              "    mov   0x8000(%[mem]), %%r15 \n"

              "    mov   0x9000(%[mem]), %%r8  \n"
              "    mov   0xa000(%[mem]), %%rcx \n"
              "    mov   0xb000(%[mem]), %%r10 \n"
              "    mov   0xc000(%[mem]), %%r11 \n"
              "    mov   0xd000(%[mem]), %%r12 \n"
              "    mov   0xe000(%[mem]), %%r13 \n"
              "    mov   0xf000(%[mem]), %%r14 \n"
              "    mov  0x10000(%[mem]), %%r15 \n")

TEST_FUNCTION(test_prefetch_no_evict, 0x80,
              "    prefetcht0   0x1000(%[mem])\n"
              "    prefetcht0   0x2080(%[mem])\n"
              "    prefetcht0   0x3100(%[mem])\n"
              "    prefetcht0   0x4180(%[mem])\n"
              "    prefetcht0   0x5200(%[mem])\n"
              "    prefetcht0   0x6280(%[mem])\n"
              "    prefetcht0   0x7300(%[mem])\n"
              "    prefetcht0   0x8380(%[mem])\n"

              "    prefetcht0   0x9400(%[mem])\n"
              "    prefetcht0   0xa480(%[mem])\n"
              "    prefetcht0   0xb500(%[mem])\n"
              "    prefetcht0   0xc580(%[mem])\n"
              "    prefetcht0   0xd600(%[mem])\n"
              "    prefetcht0   0xe680(%[mem])\n"
              "    prefetcht0   0xf700(%[mem])\n"
              "    prefetcht0  0x10780(%[mem])\n")

TEST_FUNCTION(test_prefetch_l1_evict, 0,
              "    prefetcht0   0x1000(%[mem])\n"
              "    prefetcht0   0x2000(%[mem])\n"
              "    prefetcht0   0x3000(%[mem])\n"
              "    prefetcht0   0x4000(%[mem])\n"
              "    prefetcht0   0x5000(%[mem])\n"
              "    prefetcht0   0x6000(%[mem])\n"
              "    prefetcht0   0x7000(%[mem])\n"
              "    prefetcht0   0x8000(%[mem])\n"

              "    prefetcht0   0x9000(%[mem])\n"
              "    prefetcht0   0xa000(%[mem])\n"
              "    prefetcht0   0xb000(%[mem])\n"
              "    prefetcht0   0xc000(%[mem])\n"
              "    prefetcht0   0xd000(%[mem])\n"
              "    prefetcht0   0xe000(%[mem])\n"
              "    prefetcht0   0xf000(%[mem])\n"
              "    prefetcht0  0x10000(%[mem])\n")

TEST_FUNCTION(test_store_no_evict, 0x80,
              "    movq $0,  0x1000(%[mem])\n"
              "    movq $0,  0x2080(%[mem])\n"
              "    movq $0,  0x3100(%[mem])\n"
              "    movq $0,  0x4180(%[mem])\n"
              "    movq $0,  0x5200(%[mem])\n"
              "    movq $0,  0x6280(%[mem])\n"
              "    movq $0,  0x7300(%[mem])\n"
              "    movq $0,  0x8380(%[mem])\n"

              "    movq $0,  0x9400(%[mem])\n"
              "    movq $0,  0xa480(%[mem])\n"
              "    movq $0,  0xb500(%[mem])\n"
              "    movq $0,  0xc580(%[mem])\n"
              "    movq $0,  0xd600(%[mem])\n"
              "    movq $0,  0xe680(%[mem])\n"
              "    movq $0,  0xf700(%[mem])\n"
              "    movq $0, 0x10780(%[mem])\n")

TEST_FUNCTION(test_store_l1_evict, 0,
              "    movq $0,  0x1000(%[mem])\n"
              "    movq $0,  0x2000(%[mem])\n"
              "    movq $0,  0x3000(%[mem])\n"
              "    movq $0,  0x4000(%[mem])\n"
              "    movq $0,  0x5000(%[mem])\n"
              "    movq $0,  0x6000(%[mem])\n"
              "    movq $0,  0x7000(%[mem])\n"
              "    movq $0,  0x8000(%[mem])\n"

              "    movq $0,  0x9000(%[mem])\n"
              "    movq $0,  0xa000(%[mem])\n"
              "    movq $0,  0xb000(%[mem])\n"
              "    movq $0,  0xc000(%[mem])\n"
              "    movq $0,  0xd000(%[mem])\n"
              "    movq $0,  0xe000(%[mem])\n"
              "    movq $0,  0xf000(%[mem])\n"
              "    movq $0, 0x10000(%[mem])\n")

long __attribute__((aligned(0x1000))) self_inspection(struct file *filep, unsigned int cmd, unsigned long arg) {
    struct self_inspection_data *data = (struct self_inspection_data *)arg;

    data->function_vaddresses[0] = (uint64_t)&test_load_no_evict;
    data->function_vaddresses[1] = (uint64_t)&test_load_l1_evict;
    data->function_vaddresses[2] = (uint64_t)&test_prefetch_no_evict;
    data->function_vaddresses[3] = (uint64_t)&test_prefetch_l1_evict;
    data->function_vaddresses[4] = (uint64_t)&test_store_no_evict;
    data->function_vaddresses[5] = (uint64_t)&test_store_l1_evict;

    data->energy_units = RDMSR(MSR_RAPL_POWER_UNIT);

    return 0;
}

long pf_control(struct file *filep, unsigned int cmd, unsigned long arg) {

#if !defined(IS_AMD)
    struct test_data *data = (struct test_data *)arg;

    WRMSR(MSR_MISC_FEATURE_CONTROL, data->pf_control);
#endif

    return 0;
}

long measure_energy(struct file *filep, unsigned int cmd, unsigned long arg) {

    struct test_data *data = (struct test_data *)arg;

    asm volatile(

        "    mov %[PKG], %%rcx           \n"
        "    rdmsr                       \n"
        "    mov %%rax, %[energy]        \n"

        "    mov %[PP0], %%rcx           \n"
        "    rdmsr                       \n"
        "    mov %%rax, %[energy_pp0]    \n"

        "    rdtscp                      \n"
        "    mov %%eax, %[timestamp]     \n"
        "    mov %%edx, -4+%H[timestamp] \n"

        : [ energy ] "=m"(data->energy), [ energy_pp0 ] "=m"(data->energy_pp0), [ timestamp ] "=m"(data->ticks)
        : [ PKG ] "i"(MSR_PKG_ENERGY_STATUS), [ PP0 ] "i"(MSR_PP0_ENERGY_STATUS)
        : "rax", "rcx", "rdx");

    return 0;
}

long mark_user_accessible(struct file *filep, unsigned int cmd, unsigned long arg) {

    struct mark_user_accessible_data *data = (struct mark_user_accessible_data *)arg;

    struct vm_area_struct *vma;

    uint64_t      page  = (uint64_t)data->address & ~0xFFF;
    unsigned long pages = 0;

    vma = find_vma(current->mm, page);
    if ( !vma ) {
        INFO("cannot find VMA!\n");
        return -1;
    }

    if ( data->user_accessible ) {
        pages = ptr_change_protection(vma, page, page + 0x1000, PAGE_SHARED, 0);
    }
    else {
        pages = ptr_change_protection(vma, page, page + 0x1000, PAGE_KERNEL, 0);
    }
    if ( pages != 1 ) {
        INFO("could not change page protection! %lu", pages);
        return -1;
    }

    return 0;
}

typedef long (*module_ioc_t)(struct file *filep, unsigned int cmd, unsigned long arg);

static long __attribute__((__noinline__)) module_ioctl_impl(struct file *filep, unsigned int cmd, unsigned long arg) {

    char         data[256];
    module_ioc_t handler = NULL;
    long         ret;

    if ( _IOC_SIZE(cmd) > 256 ) {
        return -EFAULT;
    }

    switch ( cmd ) {

        case TEST_LOAD_NO_EVICT:
            handler = test_load_no_evict;
            break;
        case TEST_LOAD_L1_EVICT:
            handler = test_load_l1_evict;
            break;

        case TEST_PREFETCH_NO_EVICT:
            handler = test_prefetch_no_evict;
            break;
        case TEST_PREFETCH_L1_EVICT:
            handler = test_prefetch_l1_evict;
            break;

        case TEST_STORE_NO_EVICT:
            handler = test_store_no_evict;
            break;
        case TEST_STORE_L1_EVICT:
            handler = test_store_l1_evict;
            break;

        case SELF_INSPECTION:
            handler = self_inspection;
            break;

        case MEASURE_ENERGY:
            handler = measure_energy;
            break;

        case PF_CONTROL:
            handler = pf_control;
            break;

        case MARK_USER_ACCESSIBLE:
            handler = mark_user_accessible;
            break;

        default:
            return -ENOIOCTLCMD;
    }

    if ( copy_from_user(data, (void __user *)arg, _IOC_SIZE(cmd)) )
        return -EFAULT;

    ret = handler(filep, cmd, (unsigned long)((void *)data));

    if ( !ret && (cmd & IOC_OUT) ) {
        if ( copy_to_user((void __user *)arg, data, _IOC_SIZE(cmd)) )
            return -EFAULT;
    }

    return ret;
}

//                                                         %rdi, %rsi, %rdx, %r10, %r8 and %r9.
static long __attribute__((aligned(0x1000))) module_ioctl(struct file *filep, unsigned int cmd, unsigned long arg) {
    if ( likely(cmd == DO_RSB) ) {
        asm volatile(".align 64                        \n"
                     "1:                               \n"

                     "    call 2f                      \n"
                     //"3:                               \n"
                     "    mov 0x2000(%[mem]), %%rax    \n"
                     //"    jmp 3b                       \n"
                     //"    ud2                          \n"

                     "2:                               \n"
                     "    lea 4f(%%rip), %%rax         \n"
                     "    movq %%rax, (%%rsp)          \n"
                     "    ret                          \n"
                     "4:                               \n"
                     "    nop                          \n"
                     :
                     : [ mem ] "r"(arg), "a"(0)
                     : "memory");
        return 0;
    }

    if ( likely(cmd == DO_ACCESS) ) {
        asm volatile("mov 0x2000(%[mem]), %%rax" ::[mem] "r"(arg) : "rax", "memory");
        return 0;
    }

    return module_ioctl_impl(filep, cmd, arg);
}

/********************************************************************************
 * OPEN RELEASE
 ********************************************************************************/
static int module_open(struct inode *inode, struct file *file) {
    if ( device_open_count ) {
        return -EBUSY;
    }
    INFO("opened!");

    device_open_count++;
    try_module_get(THIS_MODULE);

    return 0;
}

static int module_release(struct inode *inode, struct file *file) {
    device_open_count--;
    module_put(THIS_MODULE);
    INFO("released!");

    return 0;
}

/********************************************************************************
 * INIT EXIT
 ********************************************************************************/
int module_init_function(void) {
    INFO("Module loaded");
    if ( misc_register(&dev) ) {
        ERROR("could not register device!");
        dev.this_device = NULL;
        return -EINVAL;
    }

    direct_physical_map_address = get_direct_physical_map_address();
    ptr_change_protection       = get_change_protection_function();

    INFO("DPM @ 0x%llx\n", (long long unsigned int)direct_physical_map_address);
    INFO("change_protection @ 0x%llx\n", (long long unsigned int)ptr_change_protection);

    return 0;
}

void module_exit_function(void) {
    if ( dev.this_device ) {
        misc_deregister(&dev);
    }
    INFO("Module unloaded");
}

module_init(module_init_function);
module_exit(module_exit_function);
