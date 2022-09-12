#pragma once

#include <stdio.h>
#include <stdint.h>
#include <syscall.h>

#ifdef DBG
#define DPRINT(...) do {fprintf( stdout, __VA_ARGS__ );} while (0)
#else
#define DPRINT(...)
#endif

// This sucks. you get: "[0]: [1]: [2]: [3]: .."
#ifdef KDBGALL
#define DPRINTK(...) do {printf("[%d]: ",tid); printf( __VA_ARGS__ );} while (0)
#else
#ifdef KDBG0
#define DPRINTK(...) if(tid == 0) printf( __VA_ARGS__ );
#else
#define DPRINTK(...)
#endif
#endif

#ifdef DBG
#define ccE(err) __checkCudaErrors(err, __FILE__, __LINE__)
#else
#define ccE(err) err
#endif

inline void __checkCudaErrors(cudaError_t err, const char *file, const int line)
    {
        const char *ename;
        const char *estr;
        ename = cudaGetErrorName(err);
        estr = cudaGetErrorString(err);
        if (cudaSuccess != err)
        {
            fprintf(stderr,
                    "CUDA Runtime API error = %04d from file <%s>, line %i:\n",
                    err, file, line);
            fprintf(stderr, "\t%s : %s\n", ename, estr);
            exit(-1);
        }
    } 

#define NUM_OF_REGS 32
#define XLEN 32
typedef uint32_t REG;

enum CSTATE : uint32_t{
    RUNNING,
    EXITED,
    EBREAK,
    TRAP,
    ILG_ADDR_PC,
    ILG_MEMRD,
    ILG_MEMWR,
    BAD_INST,
    ECALL
};

// where to put a bunch of debugging funcs?
inline int32_t dumpM(){
    size_t free_memory, total_memory;
    ccE(cudaMemGetInfo(&free_memory,&total_memory));
    printf("[AVAIL: %zx]\n", free_memory);
    return free_memory;
}

typedef struct {
    CSTATE state = RUNNING;
    // this is also used to pass syscall number in case of ECALL
    // or exit value in case of EXITED
    uint32_t addr = 0x0;
} core_status_t;

__global__ void initPC(REG* pcfile, REG val);
__global__ void initSP(REG* pcfile, uint32_t addr);
extern const char* banner;

extern __global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, uint32_t maxstep);
