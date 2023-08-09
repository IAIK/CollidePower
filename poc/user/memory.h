#pragma once

extern "C" {
#include "ptedit_header.h"
}
#include "config.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

// consants
constexpr uint64_t TPH_MASK     = 1 << 17;
constexpr uint64_t HUGE_MASK    = 1 << 22;
constexpr uint64_t IS_HUGE_MASK = TPH_MASK | HUGE_MASK;
constexpr uint64_t SIZE_4kB     = 0x1000;
constexpr uint64_t SIZE_2MB     = 0x200000;
constexpr uint64_t MASK_4kB     = SIZE_4kB - 1;
constexpr uint64_t MASK_2MB     = SIZE_2MB - 1;

inline uint64_t get_ppn(uint64_t ptr) {
    static int fd = open("/proc/self/pagemap", O_RDONLY);
    if ( fd < 0 ) {
        printf("couldn't open pagemap!\n");
        exit(-1);
    }

    constexpr uint64_t mask = ((1ULL << 54) - 1);

    uint64_t value;
    if ( pread(fd, &value, sizeof(uint64_t), ptr / 0x1000 * 8) != 8 ) {
        // we couldn't find the page via the pagemap try ptedit
        return ptedit_pte_get_pfn((void *)ptr, 0);
    }
    if ( (value & mask) == 0 ) {
        // we couldn't find the page via the pagemap try ptedit
        return ptedit_pte_get_pfn((void *)ptr, 0);
    }
    return value & mask;
}

inline uint64_t get_kpage_fags(uint64_t ppn) {
    static int fd = open("/proc/kpageflags", O_RDONLY);
    if ( fd < 0 ) {
        printf("coudln't open kpageflags!\n");
        exit(-1);
    }
    uint64_t value;
    if ( pread(fd, &value, sizeof(uint64_t), ppn * 8) != 8 ) {
        printf("coudln't read page flags!\n");
        exit(-1);
    }
    return value;
}

inline bool is_huge_page(uint64_t ppn) {
    uint64_t flags = get_kpage_fags(ppn);
    return (flags & IS_HUGE_MASK) > 0;
}

inline uintptr_t get_phys(uint8_t *p) {
    uint64_t vadr = (uint64_t)p;
    uint64_t ppn  = get_ppn(vadr);

    return (ppn * 0x1000) | (vadr & (is_huge_page(ppn) ? MASK_2MB : MASK_4kB));
}

inline uint64_t get_l1_set(uint64_t phys) {
    return (phys >> 6) & 0x3F;
}

inline uint64_t get_l2_set(uint64_t phys) {
    return (phys >> 6) & 0x3FF;
}

static uint64_t get_l3_set(uint64_t phys) {
    return (phys >> 6) & 0x7FF;
}

inline bool is_huge(uint8_t *base) {
    uint64_t last = get_phys(base);

    // base not aligned?
    if ( (last & (0x1000 - 1)) != 0 )
        return false;

    // all pages must be phys contigious
    for ( size_t i = 1; i < 512; ++i ) {
        uint64_t current = get_phys(base + i * 0x1000);

        if ( last + 0x1000 != current ) {
            return false;
        }
        last = current;
    }
    return true;
}

inline uint8_t *get_huge_page() {
    uint8_t *v1 = (uint8_t *)mmap(nullptr, 2 * SIZE_2MB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
    munmap(v1, 2 * SIZE_2MB);

    v1 += SIZE_2MB - ((uint64_t)v1 % SIZE_2MB);

    uint8_t *v2 = (uint8_t *)mmap(v1, SIZE_2MB, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if ( v1 != v2 ) {
        printf("mmap failed!\n");
        exit(-1);
    }

    if ( madvise(v2, SIZE_2MB, MADV_HUGEPAGE) != 0 ) {
        printf("MADV_HUGEPAGE failed!\n");
        exit(-1);
    }
    sched_yield();
    for ( int i = 0; i < 512; ++i ) {
        v2[i * 0x1000] = i;
    }
    sched_yield();
    if ( !is_huge(v2) ) {
        printf("MADV_HUGEPAGE lied!\n");
        exit(-1);
    }

    return v2;
}

using pages_t = std::vector<uint8_t *>;

// search a page that fullfills the set requirement
inline pages_t get_pages_for_l2_sets(std::vector<uint64_t> const &sets, uint8_t *huge_page) {

    pages_t pages;
    assert(pages.size() <= 32);

    uint64_t counter = 0;
    for ( uint64_t s : sets ) {
        pages.push_back(huge_page + s * 64 + (counter++ << 16));
    }

    return pages;
}

// map one vadr to another
inline void map_vadr_to_vadr(uint8_t *vadr_from, uint8_t *vadr_to) {
    size_t offset = (get_phys(vadr_from) / 0x1000) * 0x1000;

    uint8_t *p = (uint8_t *)mmap(vadr_to, 0x1000, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ptedit_umem, offset);
    if ( p == MAP_FAILED ) {
        printf("mmap failed!\n");
        exit(-1);
    }

    sched_yield();
    sched_yield();
    sched_yield();

    *p = rand();

    if ( p != vadr_to || *vadr_to != *vadr_from ) {
        printf("Error mapping!\n");
        exit(-1);
    }

    if ( get_ppn((uint64_t)vadr_from) != get_ppn((uint64_t)vadr_to) ) {
        printf("Error padr not matching %lx %lx!\n", get_ppn((uint64_t)vadr_from), get_ppn((uint64_t)vadr_to));
        exit(-1);
    }
}

// allocate the memory for the experiment
inline pages_t prepare_memory(Evict evict, uint8_t *huge_page) {

    std::vector<uint64_t> l2_sets;

    switch ( evict ) {
        case Evict::NONE:
            // No  eviction is implemented at the mov/prefetch instructions
            // use same memory as L1 eviction
        case Evict::L1:
            fprintf(stderr, "# allocating L1E memory:\n");
            l2_sets = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0 };
            break;
        case Evict::L1_L2:
            fprintf(stderr, "# allocating L12E memory:\n");
            l2_sets = { 10, 10, 10, 10, 10, 10, 10, 10, 1, 2, 3, 4, 5, 6, 7, 8, 10 };
            break;
    }

    // shift the bits up to the l2 position
    for ( uint64_t &s : l2_sets ) {
        s = s << 6;
    }

    pages_t pages = get_pages_for_l2_sets(l2_sets, huge_page);

    bool all_l1 = std::all_of(pages.begin(), pages.end(), [&](uint8_t *v) {
        return get_l1_set(get_phys(pages.front())) == get_l1_set(get_phys(v));
    });

    if ( !all_l1 ) {
        printf("Error not all pages are in the same l1 set ... how?!\n");
        exit(-1);
    }

    for ( uint8_t *p : pages ) {
        uint64_t padr_from = get_phys(p);
        fprintf(stderr, "# allocated: 0x%lx -> 0x%lx [%5ld|%5ld|%5ld]\n", (uint64_t)p, padr_from, get_l1_set(padr_from), get_l2_set(padr_from), get_l3_set(padr_from));
    }

    return pages;
}

inline void map_pages(uint8_t *base, pages_t const &pages) {

    for ( size_t i = 0; i < pages.size(); ++i ) {
        if ( munmap(base + 0x1000 * i, 0x1000) != 0 ) {
            printf("error mem unmap!\n");
        }
    }

    for ( size_t i = 0; i < pages.size(); ++i ) {
        *(pages[i] + 0x1000 / 2) = i;
        map_vadr_to_vadr(pages[i], base + 0x1000 * i);
    }

    memset(base, 0, 0x1000 * pages.size());
    for ( uint64_t i = 0; i < pages.size(); ++i ) {
        base[i * 0x1000 + 0x1000 / 2] = i;
    }
}

// struct to manage memory
struct Memory {
    uint8_t *base;
    uint8_t *huge_page;

    pages_t l1e;
    pages_t l12e;

    Memory(uint8_t *_base) {
        base      = _base;
        huge_page = get_huge_page();

        fprintf(stderr, "# base: 0x%lx\n", (uint64_t)_base);

        l1e  = prepare_memory(Evict::L1, huge_page);
        l12e = prepare_memory(Evict::L1_L2, huge_page);
    }

    uint8_t *get_memory(Evict x, uint64_t l1_set) const {
        assert((l1_set & ~0x3f) == 0);
        // shift the memory region to the correct L1 set
        switch ( x ) {
            case Evict::NONE:
                map_pages(base, l1e);
                return base;
            case Evict::L1:
                map_pages(base, l1e);
                return base + l1_set * 64;
            case Evict::L1_L2:
                map_pages(base, l12e);
                return base + l1_set * 64;
        }
        assert(false);
        __builtin_unreachable();
    }
};
