#include "dbg.cuh"

// supress warning: 'long double' is treated as 'double' in device code
#pragma nv_diag_suppress 20208

// For MMU?
extern __constant__ uint32_t core_mem_size;
extern __constant__ uint32_t cro_size;
extern __constant__ uint8_t cmem[CRO_MAX_SIZE];

// #######################################
// ###  Instruction Formats: IBJRSU
// #######################################
#define FMT_I                           \
    rd = (insn >> 7) & 0x1f;   \
    rs1 = (insn >> 15) & 0x1f; \
    imm = (insn & 0x80000000 ? 0xfffff800 : 0) | ((insn >> 20) & 0x000007ff);

#define FMT_B                           \
    rd = (insn >> 7) & 0x1f;   \
    rs2 = (insn >> 20) & 0x1f; \
    rs1 = (insn >> 15) & 0x1f; \
    imm = (insn & 0x80000000 ? 0xfffff000 : 0) | ((insn << 4) & 0x00000800) | ((insn >> 20) & 0x000007e0) | ((insn >> 7) & 0x0000001e);

#define FMT_J                         \
    rd = (insn >> 7) & 0x1f; \
    imm = (insn & 0x80000000 ? 0xfff00000 : 0) | (insn & 0x000ff000) | ((insn & 0x00100000) >> 9) | ((insn & 0x7fe00000) >> 20);

#define FMT_R                           \
    rd = (insn >> 7) & 0x1f;   \
    rs1 = (insn >> 15) & 0x1f; \
    rs2 = (insn >> 20) & 0x1f; \
    rs3 = (insn >> 27) & 0x1f;

#define FMT_S                           \
    rs1 = (insn >> 15) & 0x1f; \
    rs2 = (insn >> 20) & 0x1f; \
    imm = (insn & 0x80000000 ? 0xfffff000 : 0) | ((insn >> 20) & 0xfe0) | ((insn >> 7) & 0x1f);

#define FMT_U                         \
    rd = (insn >> 7) & 0x1f; \
    imm = insn & 0xfffff000;

#define INC_PC pc = pc + 4;

// sign extend
__device__ inline uint32_t sxtn(uint32_t x, uint32_t b){
    uint32_t m = ((uint)1) << (b - 1);
    return (x ^ m) - m;
}

__global__ void dumpS(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec, int coreid) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid != 0) return;
    printf("[%d]: ", coreid);   
}