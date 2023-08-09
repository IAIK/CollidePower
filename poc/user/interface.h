#pragma once

#include "../module/interface.h"
#include "config.h"
#include "memory.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

extern int fd;

inline test_data measure() {
    test_data data;
    memset(&data, 0, sizeof(test_data));
    if ( ioctl(fd, MEASURE_ENERGY, &data) < 0 ) {
        printf("ioctl error\n");
        exit(-1);
    }
    return data;
}

inline void set_pf_control(HWPF pf_control) {
    test_data data;
    memset(&data, 0, sizeof(test_data));
    data.pf_control = static_cast<uint8_t>(pf_control);
    if ( ioctl(fd, PF_CONTROL, &data) < 0 ) {
        printf("ioctl error\n");
        exit(-1);
    }
}

inline void mark_user_accessible(void *address, bool user_accessible) {
    mark_user_accessible_data data;
    memset(&data, 0, sizeof(mark_user_accessible_data));
    data.address         = (uint64_t)address;
    data.user_accessible = (uint64_t)user_accessible;

    if ( ioctl(fd, MARK_USER_ACCESSIBLE, &data) < 0 ) {
        printf("ioctl error\n");
        exit(-1);
    }
}

inline void print_fields(char prefix) {
    fprintf(stderr, ",%cEnergy,%cEnergyPP0,%cTicks,%cPState,%cTemp,%cVolt,%cAPerf,%cMperf,%cCalib", prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix, prefix);
}

inline void print_data(test_data const &x) {
    fprintf(stderr, ",%6lld,%6lld,%6lld,%6lld,%6lld,%6lld,%6lld,%6lld,%6lld", x.energy, x.energy_pp0, x.ticks, x.pstate, x.temperature, x.voltage, x.aperf, x.mperf, x.calibration);
}

inline void self_inspection() {
    self_inspection_data data;
    memset(&data, 0, sizeof(self_inspection_data));

    if ( ioctl(fd, SELF_INSPECTION, &data) < 0 ) {
        printf("ioctl error\n");
        exit(-1);
    }

    fprintf(stderr, "# Self Inspection Data:\n");

    for ( size_t i = 0; i < 6; ++i ) {
        uint8_t *vadr = (uint8_t *)data.function_vaddresses[i];

        uint64_t p0 = get_phys(vadr + 0x0000);
        uint64_t p1 = get_phys(vadr + 0x1000);
        uint64_t p2 = get_phys(vadr + 0x2000);

        fprintf(stderr, "# Function: %lu vadr: 0x%16lx padrs: 0x%16lx|0x%16lx|0x%16lx l2_sets: %6lu|%6lu|%6lu\n", i, (uint64_t)vadr, p0, p1, p2, get_l2_set(p0), get_l2_set(p1),
                get_l2_set(p2));
    }

    uint32_t energy_units = (data.energy_units & 0x1f00) >> 8;

    printf("Energy Unit: %f uJ\n", 1000000.0 / (1 << energy_units));
}