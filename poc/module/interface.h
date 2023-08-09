#pragma once

#include <linux/ioctl.h>
#include <linux/types.h>

#define MAGIC 'P'

#define TEST_LOAD_NO_EVICT _IOWR(MAGIC, 1, struct test_data)
#define TEST_LOAD_L1_EVICT _IOWR(MAGIC, 2, struct test_data)

#define TEST_PREFETCH_NO_EVICT _IOWR(MAGIC, 5, struct test_data)
#define TEST_PREFETCH_L1_EVICT _IOWR(MAGIC, 6, struct test_data)

#define TEST_STORE_NO_EVICT _IOWR(MAGIC, 8, struct test_data)
#define TEST_STORE_L1_EVICT _IOWR(MAGIC, 9, struct test_data)

#define SELF_INSPECTION _IOWR(MAGIC, 12, struct self_inspection_data)

#define DO_ACCESS      _IOWR(MAGIC, 13, struct test_data)
#define DO_RSB         _IOWR(MAGIC, 14, struct test_data)
#define MEASURE_ENERGY _IOWR(MAGIC, 16, struct test_data)
#define PF_CONTROL     _IOWR(MAGIC, 17, struct test_data)

#define MARK_USER_ACCESSIBLE _IOWR(MAGIC, 18, struct mark_user_accessible_data)

struct test_data {
    __u64 address;
    __u64 count;
    __u64 pf_control;

    __u64 energy;
    __u64 energy_pp0;
    __u64 ticks;
    __u64 voltage;
    __u64 pstate;
    __u64 temperature;
    __u64 aperf;
    __u64 mperf;
    __u64 calibration;
} __attribute__((__packed__));

struct self_inspection_data {
    __u64 function_vaddresses[6];
    __u64 energy_units;
} __attribute__((__packed__));

struct mark_user_accessible_data {
    __u64 address;
    __u64 user_accessible;
} __attribute__((__packed__));
