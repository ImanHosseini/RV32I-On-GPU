#pragma once

#include <stdio.h>
#include <stdint.h>

#ifdef DBG
#define DPRINT(...) do {fprintf( stdout, __VA_ARGS__ );} while (0)
#else
#define DPRINT(...)
#endif

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
    HLT,
    TRAP,
    ILG,
    ECALL
};

typedef struct {
    CSTATE state = RUNNING;
    uint32_t addr = 0x0;
} core_status_t;

__global__ void initPC(REG* pcfile, REG val);
extern const char* banner;
