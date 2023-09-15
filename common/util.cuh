#pragma once

#include <stdio.h>
#include <stdint.h>
#include <syscall.h>
#include <string>
#include <vector>
#include <sstream>

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

#ifndef CRO_MAX_SIZE
#define CRO_MAX_SIZE 8*1024
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

#define CSTATE_ENUM_VALUES \
    X(RUNNING)              \
    X(MAXSTEP)              \
    X(EXITED)               \
    X(EBREAK)               \
    X(TRAP)                 \
    X(ILG_ADDR_PC)          \
    X(ILG_MEMRD)            \
    X(ILG_MEMWR_OB)         \
    X(ILG_MEMWR_RO)         \
    X(BAD_INST)             \
    X(ECALL)                \
    X(BKPT)

// Create the enum
enum CSTATE : uint32_t {
#define X(name) name,
    CSTATE_ENUM_VALUES
#undef X
    CSTATE_COUNT
};

extern const char* CSTATE_STRINGS[];
const char* CStateToString(CSTATE state);

// where to put a bunch of debugging funcs?
inline int32_t dumpM(){
    size_t free_memory, total_memory;
    ccE(cudaMemGetInfo(&free_memory,&total_memory));
    printf("[AVAIL: %zx]\n", free_memory);
    return free_memory;
}

inline std::vector<std::string> split(const std::string& s, char delimiter = ' ')
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

typedef struct {
    CSTATE state = MAXSTEP;
    // this is also used to pass syscall number in case of ECALL
    // or exit value in case of EXITED
    uint32_t addr = 0x0;
} core_status_t;

__global__ void initPC(REG* pcfile, REG val);
__global__ void initSP(REG* pcfile, uint32_t addr);
extern const char* banner;

#ifdef DBGX
extern __global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, uint32_t maxstep = 0, uint32_t* bkpts = nullptr, int num_bkpts = 0);
#else
#ifndef COV
extern __global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, uint32_t maxstep = 0);
#else
extern void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, uint32_t maxstep, uint16_t* bhshmap, uint8_t* gcovmap);
#endif
#endif

extern __global__ void dumpS(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, int coreid, bool* pop_bt);
