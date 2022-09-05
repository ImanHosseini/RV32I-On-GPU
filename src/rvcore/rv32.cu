#include "util.cuh"

// RISC-V impl is based on https://github.com/PiMaker/rvc/tree/master/src

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

// #######################################
// ###  Memory Access Funcs
// #######################################
#ifdef MMU
__device__ bool read8(uint32_t addr, uint8_t& value){
    // Not implemented yet!
}
#else
__device__ bool read8(uint32_t addr, uint8_t& value, core_status_t& cstatus){
    return;
}
#endif

// For MMU?
// __constant__ int page_cnt;
// __constant__ int page_size;
__constant__ uint32_t core_mem_size;
// __constant__ uint32_t core_per_grid;
#define MEMORY_SLACK 1024*4

void set_cms(uint32_t* cms){
    printf("V:%u\n",*cms);
    ccE(cudaMemcpyToSymbol(core_mem_size, cms, sizeof(uint32_t)));
}

// void initialize(REG* &regfile, REG* &pcfile, core_status_t* &svec, cudaStream_t* streams, int np, int nq){
//     size_t free_memory, total_memory;
//     ccE(cudaMemGetInfo(&free_memory,&total_memory));
//     free_memory -= MEMORY_SLACK;
//     uint32_t cms = (free_memory/ np) - (sizeof(REG)+1)*NUM_OF_REGS - sizeof(core_status_t);
//     printf("|Q: %d|Np: %d|Mem/Core: 0x%x|\n", nq, np, cms);
//     ccE(cudaMemcpyToSymbol(core_mem_size, &cms, sizeof(uint32_t)));
//     // ccE(cudaMalloc());
// }

void qinit(REG* &regfile, REG* &pcfile, core_status_t* &svec, cudaStream_t s, int mpc)
{
    
}

uint32_t calc_mpc(int np, int nq){
    size_t free_memory, total_memory;
    ccE(cudaMemGetInfo(&free_memory,&total_memory));
    free_memory -= MEMORY_SLACK;
    uint32_t mpc = (free_memory/ np) - (sizeof(REG)+1)*NUM_OF_REGS - sizeof(core_status_t);
    mpc = mpc - (mpc%256);
    printf("|Q: %d|Np: %d|Mem/Core: 0x%x|\n", nq, np, mpc);
    ccE(cudaMemcpyToSymbol(core_mem_size, &mpc, sizeof(uint32_t)));
    return mpc;
}

__device__ inline void stepN(REG *regs, REG& pc, uint8_t *mem, int tid, core_status_t& cstatus)
{
    // int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // REG *regs = regfile + tid * NUM_OF_REGS;
    regs[0] = 0;
    // what if pc accesses out of bounds memory?
    uint32_t rd, rs1, rs2, rs3, imm;
    // REG &pc = *(pcfile + tid);
    uint32_t insn = *((uint32_t *)(mem + pc));
    uint32_t insn_masked = insn & 0x0000007f;
    switch (insn_masked)
    {
    case 0x00000017:
    {
        // RV32I auipc
        FMT_U
        DPRINTK("auipc x%d, 0x%x\n", rd, imm >> 12);
        regs[rd] = pc + imm;
        INC_PC
        return;
    }
    case 0x0000006f:
    {
        // RV32I jal
        FMT_J
        DPRINTK("jal x%d, 0x%x\n", rd, pc + imm);
        regs[rd] = pc + 4;
        pc = pc + imm;
        return;
    }
    case 0x00000037:
    {
        // RV32I lui
        FMT_U
        DPRINTK("lui x%d, 0x%x\n", rd, imm >> 12);
        regs[rd] = imm;
        INC_PC
        return;
    }
    default:
    {
        // DPRINTK("default 0x%x\n",insn);
        // INC_PC
    }
    }
    // R-type instructions
    {
        FMT_R
        insn_masked = insn & 0xfe00707f;
        switch (insn_masked)
        {
        case 0x00000033:
        {
            // RV32I add
            DPRINTK("add x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = (int32_t)regs[rs1] + (int32_t)regs[rs2];
            break;
        }
        case 0x00007033:
        {
            // RV32I and
            DPRINTK("and x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = (int32_t)regs[rs1] & (int32_t)regs[rs2];
            break;
        }
        case 0x02004033:
        {
            // RV32M div
            DPRINTK("div x%d, x%d, x%d\n", rd, rs1, rs2);
            REG dividend = regs[rs1];
            REG divisor = regs[rs2];
            REG res;
            if (divisor == 0)
            {
                res = 0xFFFFFFFF;
            }
            else if (dividend == 0x80000000 && divisor == 0xFFFFFFFF)
            {
                res = dividend;
            }
            else
            {
                int32_t tmp = (int32_t)dividend / (int32_t)divisor;
                res = uint32_t(tmp);
            }
            regs[rd] = res;
            break;
        }
        case 0x02005033:
        {
            // RV32M divu
            DPRINTK("divu x%d, x%d, x%d\n", rd, rs1, rs2);
            REG dividend = regs[rs1];
            REG divisor = regs[rs2];
            REG res;
            if (divisor == 0)
            {
                res = 0xFFFFFFFF;
            }
            else
            {
                res = dividend / divisor;
            }
            regs[rd] = res;
            break;
        }
        case 0x02000033:
        {
            // RV32M mul
            DPRINTK("mul x%d, x%d, x%d\n", rd, rs1, rs2);
            uint32_t tmp = (int32_t)regs[rs1] / (int32_t)regs[rs2];
            regs[rd] = tmp;
            break;
        }
        case 0x02001033:
        {
            // RV32M mulh
            // see: https://github.com/PiMaker/rvc/blob/9918aea234fe8f1536fc0dd237de6709f378c9db/_Nix/rvc/src/emu.h#L257
            DPRINTK("mulh x%d, x%d, x%d\n", rd, rs1, rs2);
            double op1 = (int32_t)regs[rs1];
            double op2 = (int32_t)regs[rs2];
            uint32_t tmp = (uint32_t)((op1 * op2) / 4294967296.0l); // '/ 4294967296' == '>> 32'
            regs[rd] = tmp;
            break;
        }
        case 0x02002033:
        {
            // RV32M mulhsu
            DPRINTK("mulhsu x%d, x%d, x%d\n", rd, rs1, rs2);
            double op1 = (int32_t)regs[rs1];
            double op2 = (uint32_t)regs[rs2];
            uint32_t tmp = (uint32_t)((op1 * op2) / 4294967296.0l); // '/ 4294967296' == '>> 32'
            regs[rd] = tmp;
            break;
        }
        case 0x02003033:
        {
            // RV32M mulhu
            DPRINTK("mulhu x%d, x%d, x%d\n", rd, rs1, rs2);
            double op1 = (uint32_t)regs[rs1];
            double op2 = (uint32_t)regs[rs2];
            uint32_t tmp = (uint32_t)((op1 * op2) / 4294967296.0l); // '/ 4294967296' == '>> 32'
            regs[rd] = tmp;
            break;
        }
        case 0x00006033:
        {
            // RV32I or
            DPRINTK("or x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = regs[rs1] | regs[rs2];
            break;
        }
        case 0x02006033:
        {
            // RV32M rem
            DPRINTK("rem x%d, x%d, x%d\n", rd, rs1, rs2);
            REG dividend = regs[rs1];
            REG divisor = regs[rs2];
            REG res;
            if (divisor == 0)
            {
                res = dividend;
            }
            else if (dividend == 0x80000000 && divisor == 0xFFFFFFFF)
            {
                res = 0;
            }
            else
            {
                int32_t tmp = (int32_t)dividend % (int32_t)divisor;
                res = uint32_t(tmp);
            }
            regs[rd] = res;
            break;
        }
        case 0x02007033:
        {
            // RV32M remu
            DPRINTK("remu x%d, x%d, x%d\n", rd, rs1, rs2);
            REG dividend = regs[rs1];
            REG divisor = regs[rs2];
            REG res;
            if (divisor == 0)
            {
                res = dividend;
            }
            else
            {
                res = dividend % divisor;
            }
            regs[rd] = res;
            break;
        }
        case 0x00001033:
        {
            // RV32I sll
            DPRINTK("sll x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = regs[rs1] << regs[rs2];
            break;
        }
        case 0x00002033:
        {
            // RV32I slt
            DPRINTK("slt x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = ((int32_t)regs[rs1] < (int32_t)regs[rs2]) ? 1 : 0;
            break;
        }
        case 0x00003033:
        {
            // RV32I sltu
            DPRINTK("sltu x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = ((uint32_t)regs[rs1] < (uint32_t)regs[rs2]) ? 1 : 0;
            break;
        }
        case 0x40005033:
        {
            // RV32I sra
            DPRINTK("sra x%d, x%d, x%d\n", rd, rs1, rs2);
            uint32_t msr = regs[rs1] & 0x80000000;
            regs[rd] = msr ? ~(~regs[rs1] >> regs[rs2]) : regs[rs1] >> regs[rs2];
            break;
        }
        case 0x00005033:
        {
            // RV32I srl
            DPRINTK("srl x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = regs[rs1] >> regs[rs2];
            break;
        }
        case 0x40000033:
        {
            // RV32I sub
            DPRINTK("sub x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = (int32_t)regs[rs1] - (int32_t)regs[rs2];
            break;
        }
        case 0x00004033:
        {
            // RV32I xor
            DPRINTK("xor x%d, x%d, x%d\n", rd, rs1, rs2);
            regs[rd] = regs[rs1] ^ regs[rs2];
            break;
        }
        default:
        {
            goto other;
        }
        }
        INC_PC
        return;
    }

other:
    insn_masked = insn & 0x0000707f;
    switch (insn_masked)
    {
    case 0x00000013:{
        // RV32I addi
        FMT_I
        DPRINTK("addi x%d, x%d, 0x%x\n", rd, rs1, imm);
        regs[rd] = (int32_t) regs[rs1] + (int32_t) imm;
        INC_PC
        return;
    }
    case 0x00007013:{
        // RV32I andi
        FMT_I
        DPRINTK("andi x%d, x%d, 0x%x\n", rd, rs1, imm);
        regs[rd] = regs[rs1] & imm;
        INC_PC
        return;
    }
    case 0x00000063:{
        // RV32I beq
        FMT_B
        DPRINTK("beq x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
        if(regs[rs1] == regs[rs2]) {
            pc = pc + imm;
            return;
        }
        INC_PC
        return;
    }
    case 0x00005063:{
        // RV32I bge
        FMT_B
        DPRINTK("bge x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
        if((int32_t) regs[rs1] >= (int32_t) regs[rs2]) {
            pc = pc + imm;
            return;
        }
        INC_PC
        return;
    }
    case 0x00007063:{
        // RV32I bgeu
        FMT_B
        DPRINTK("bgeu x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
        if(regs[rs1] >= regs[rs2]) {
            pc = pc + imm;
            return;
        }
        INC_PC
        return;
    }
    case 0x00004063:{
        // RV32I blt
        FMT_B
        DPRINTK("blt x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
        if((int32_t) regs[rs1] < (int32_t) regs[rs2]) {
            pc = pc + imm;
            return;
        }
        INC_PC
        return;
    }
    case 0x00006063:{
        // RV32I bltu
        FMT_B
        DPRINTK("bltu x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
        if(regs[rs1] < regs[rs2]) {
            pc = pc + imm;
            return;
        }
        INC_PC
        return;
    }
    case 0x00001063:{
        // RV32I bne
        FMT_B
        DPRINTK("bne x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
        if(regs[rs1] != regs[rs2]) {
            pc = pc + imm;
            return;
        }
        INC_PC
        return;
    }
    case 0x00000067:{
        // RV32I jalr
        FMT_I
        DPRINTK("jalr x%d, x%x, 0x%x\n", rd, rs1, imm);
        regs[rd] = pc + 4;
        pc = regs[rs1] + imm;
        return;
    }
    case 0x00000003:{
        // RV32I lb
        FMT_I
        DPRINTK("lb x%d, x%x, 0x%x\n", rd, rs1, imm);
        // MEM-ACCESS
        return;
    }
    default:{

    }
    }

    return;
}

__launch_bounds__(32)
__global__ void step(REG *regfile, REG *pcfile, uint8_t *gmem, core_status_t *svec){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("CMS:%d\n",core_mem_size);
    REG *regs = regfile + tid * NUM_OF_REGS;
    // what if pc accesses out of bounds memory?
    REG &pc = *(pcfile + tid);
    uint8_t* mem = gmem + tid * core_mem_size; 
    core_status_t& cstatus = *(svec + tid);  
    stepN(regfile, pc, mem, tid, cstatus);
}