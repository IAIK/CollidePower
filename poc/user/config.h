#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <openssl/sha.h>
#include <random>

// number of samples
constexpr size_t NUMBER_SAMPLES = 300000;

// repeat per randomly choosen parameter, counteract dynamic states
constexpr size_t EXPERIMENT_REPEAT = 100;
constexpr size_t SAMPLE_REPEAT     = 1;

// number of inner loop iterations = amplification factor
constexpr size_t LOOPS = 0x5000000 / 10;

// instruction used
enum class Inst {
    // Signal
    LOAD,
    PREFETCH,
    STORE,

    // Meltdown-Power
    LOAD_MELTDOWN,
    STORE_MELTDOWN,

    // MDS-Power
    LOAD_MDS,
    STORE_MDS,
};

// eviction type
enum class Evict { NONE, L1, L1_L2 };

// enable or disable hardware prefetchers
enum class HWPF : uint8_t { DISABLED = 0xf, ENABLED = 0x0 };

// type of cache line fill to use
enum class CLFill : uint8_t { ZEROS = 0, DISTINCT = 0x1, RANDOM = 0x2, CONSTANT = 0x3, DISTINCT_ADJACENT = 0x4 };

// name utils
constexpr inline char const *name(Inst x) {
    switch ( x ) {
        case Inst::LOAD:
            return "LD";
        case Inst::STORE:
            return "ST";
        case Inst::PREFETCH:
            return "PF";

        case Inst::LOAD_MELTDOWN:
            return "LD_MELTDOWN";
        case Inst::STORE_MELTDOWN:
            return "ST_MELTDOWN";

        case Inst::LOAD_MDS:
            return "LD_MDS";
        case Inst::STORE_MDS:
            return "ST_MDS";
    }
    assert(false);
    __builtin_unreachable();
}

constexpr inline char const *name(Evict x) {
    switch ( x ) {
        case Evict::NONE:
            return "NOE";
        case Evict::L1:
            return "L1E";
        case Evict::L1_L2:
            return "L12E";
    }
    assert(false);
    __builtin_unreachable();
}

constexpr inline char const *name(HWPF x) {
    switch ( x ) {
        case HWPF::DISABLED:
            return "DPF";
        case HWPF::ENABLED:
            return "EPF";
    }
    assert(false);
    __builtin_unreachable();
}

constexpr inline char const *name(CLFill x) {
    switch ( x ) {
        case CLFill::ZEROS:
            return "Z";
        case CLFill::DISTINCT:
            return "D";
        case CLFill::RANDOM:
            return "R";
        case CLFill::CONSTANT:
            return "C";
        case CLFill::DISTINCT_ADJACENT: // same as DISTINCT for post processing
            return "D";
    }
    assert(false);
    __builtin_unreachable();
}

// struct to represent config
struct ExperimentConfig {
    uint64_t id;
    uint64_t nibble_begin;
    uint64_t nibble_end;
    uint64_t nibble_stride;
    Inst     instruction;
    Evict    eviction;
    uint64_t l1_set;
    HWPF     pf_control;
    CLFill   v_fill;
    CLFill   g_fill;
    double   sample_scale;

    char name[100];

    ExperimentConfig(uint64_t id, uint64_t nibble_begin, uint64_t nibble_end, uint64_t nibble_stride, Inst instruction, Evict eviction, uint64_t l1_set, HWPF pf_control,
                     CLFill v_fill, CLFill g_fill, double sample_scale)
      : id { id }
      , nibble_begin { nibble_begin }
      , nibble_end { nibble_end }
      , nibble_stride { nibble_stride }
      , instruction { instruction }
      , eviction { eviction }
      , l1_set { l1_set }
      , pf_control { pf_control }
      , v_fill { v_fill }
      , g_fill { g_fill }
      , sample_scale { sample_scale }
      , name {}

    {
        snprintf(name, sizeof(name), "%02lu_n%02lu_n%02lu_n%02lu_%s_%s_%s_%s_%s_%02lu", id, nibble_begin, nibble_end, nibble_stride, ::name(instruction), ::name(eviction),
                 ::name(pf_control), ::name(v_fill), ::name(g_fill), l1_set);
    }

    // print csv header
    static void print_header() {
        fprintf(stderr, ",Exp,Id,NBegin,NEnd,NStride,Instruction,Eviction,PFControl,L1Set,MemIndex,CLVFill,CLGFill");
    }

    // print config as row
    void print_config(uint8_t mem_index) const {
        fprintf(stderr, ",%s,%02lu,%02lu,%02lu,%02lu,%s,%s,%s,%02lu,%02u,%s,%s", name, id, nibble_begin, nibble_end, nibble_stride, ::name(instruction), ::name(eviction),
                ::name(pf_control), l1_set, mem_index, ::name(v_fill), ::name(g_fill));
    }
};

// struct to manage the fill value of cache lines
struct CLFillValue {
    uint64_t seed;
    bool     is_inverted;
    CLFill   type;

    CLFillValue(CLFill fill_type, std::random_device &rd)
      : seed { 0 }
      , is_inverted { false }
      , type { fill_type } {

        std::uniform_int_distribution<uint64_t> dist_random(0, ~(uint32_t)0);
        std::uniform_int_distribution<uint64_t> dist_nibble(0, 15);

        switch ( fill_type ) {
            case CLFill::CONSTANT:
            case CLFill::ZEROS:
                seed = dist_nibble(rd);
                break;
            case CLFill::DISTINCT:
            case CLFill::DISTINCT_ADJACENT:
                seed = (dist_nibble(rd) << 4) | dist_nibble(rd);
                break;
            case CLFill::RANDOM:
                seed = dist_random(rd);
                break;
        }
    }

    CLFillValue invert() const {
        CLFillValue copy = *this;
        copy.is_inverted = !is_inverted;
        return copy;
    }

    // fill mem with nibble
    static void fill_nibble(uint8_t *mem, uint64_t nibble_index, uint8_t nibble_value) {
        uint64_t byte  = nibble_index / 2;
        uint64_t shift = nibble_index & 1;

        mem[byte] &= ~(0xF << (shift * 4));
        mem[byte] |= (nibble_value & 0xF) << (shift * 4);
    }

    void fill_cl_constant_real(ExperimentConfig const &c, uint8_t *mem, uint8_t real) {

        // fill with "real"
        for ( uint64_t nibble_index = c.nibble_begin; nibble_index < c.nibble_end; nibble_index += c.nibble_stride ) {
            fill_nibble(mem, nibble_index, real ^ (is_inverted ? 0xF : 0));
        }
    }

    void fill_cl_constant_fill(ExperimentConfig const &c, uint8_t *mem, uint8_t fill) {
        // fill with "fill"
        for ( uint64_t nibble_index = 0; nibble_index < c.nibble_begin; nibble_index += c.nibble_stride ) {
            fill_nibble(mem, nibble_index, fill ^ (is_inverted ? 0xF : 0));
        }

        for ( uint64_t nibble_index = c.nibble_end; nibble_index < 128; nibble_index += c.nibble_stride ) {
            fill_nibble(mem, nibble_index, fill ^ (is_inverted ? 0xF : 0));
        }
    }

    void fill_cl(ExperimentConfig const &c, uint8_t *mem) {
        // zero cl
        memset(mem, 0, 128);

        switch ( type ) {

            case CLFill::ZEROS:
                fill_cl_constant_real(c, mem, (seed >> 0) & 0xF);
                break;
            case CLFill::DISTINCT:
                fill_cl_constant_real(c, mem, (seed >> 0) & 0xF);
                fill_cl_constant_fill(c, mem, (seed >> 4) & 0xF);
                break;
            case CLFill::DISTINCT_ADJACENT:
                if ( ((uint64_t)mem & 0xFFF) != 0 ) {
                    printf("the memory is not aligned, adjacent CLs might not work!\n");
                    exit(-3);
                }
                fill_cl_constant_real(c, mem + 0, (seed >> 0) & 0xF);
                fill_cl_constant_real(c, mem + 64, (seed >> 4) & 0xF);
                break;

            case CLFill::RANDOM:
                SHA512((unsigned char *)&seed, sizeof(uint64_t), (unsigned char *)mem);
                if ( is_inverted ) {
                    for ( uint64_t i = 0; i < 64; ++i ) {
                        mem[i] = mem[i] ^ 0xFF;
                    }
                }
                break;
            case CLFill::CONSTANT: {
                assert(is_inverted == false);
                assert(c.nibble_begin == 0);
                assert(c.nibble_end == 128);
                uint64_t constant_seed = 0x1234;
                SHA512((unsigned char *)&constant_seed, sizeof(uint64_t), (unsigned char *)mem);
                fill_cl_constant_real(c, mem, (seed >> 0) & 0xF);
                break;
            }
        }

        // prevent zero store optimization
        // we never actively target this bit
        // diff meas will remove this effect in all other cases
        // mem[31] |= 1;
        // mem[63] |= 1;

        if constexpr ( false ) {
            uint64_t *p = (uint64_t *)(mem);
            printf("[0x%lx] = 0x%lx 0x%lx 0x%lx 0x%lx 0x%lx 0x%lx 0x%lx 0x%lx\n", (uint64_t)p, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
            p = (uint64_t *)(mem + 64);
            printf("[0x%lx] = 0x%lx 0x%lx 0x%lx 0x%lx 0x%lx 0x%lx 0x%lx 0x%lx\n", (uint64_t)p, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
        }
    }
};