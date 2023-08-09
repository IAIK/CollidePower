#pragma once

#include <assert.h>
#include <linux/perf_event.h>
#include <setjmp.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
uint64_t rdtsc() {
    uint64_t a, d;
    asm volatile("mfence");
    asm volatile("rdtsc" : "=a"(a), "=d"(d));
    a = (d << 32) | a;
    asm volatile("mfence");
    return a;
}

// ---------------------------------------------------------------------------
void flush(void *p) {
    asm volatile("clflush 0(%0)\n" : : "c"(p) : "rax");
}

// ---------------------------------------------------------------------------
void maccess(void *p) {
    asm volatile("movq (%0), %%rax\n" : : "c"(p) : "rax");
}

// ---------------------------------------------------------------------------
void mfence() {
    asm volatile("mfence");
}

// ---------------------------------------------------------------------------
void nospec() {
    asm volatile("lfence");
}

// ---------------------------------------------------------------------------
int flush_reload_t(void *ptr) {
    uint64_t start = 0, end = 0;

    start = rdtsc();
    maccess(ptr);
    end = rdtsc();

    mfence();

    flush(ptr);

    return (int)(end - start);
}
