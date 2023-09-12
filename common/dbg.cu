#include <dbg.cuh>

__device__ __host__ void dumpInstruction(uint32_t insn) {
    uint32_t opcode = insn & 0x7f;
    uint32_t rs1 = (insn >> 15) & 0x1f;
    uint32_t rs2 = (insn >> 20) & 0x1f;
    uint32_t rd = (insn >> 7) & 0x1f;
    uint32_t funct3 = (insn >> 12) & 0x7;
    uint32_t funct7 = (insn >> 25) & 0x7f;
    uint32_t imm_i = (insn >> 20) & 0xfff;
    uint32_t imm_s = ((insn >> 7) & 0x1f) | ((insn >> 20) & 0xfe0);
    uint32_t imm_b = ((insn >> 8) & 0xf) | ((insn << 4) & 0x1000) | ((insn >> 19) & 0x7e0) | ((insn >> 20) & 0x800);
    uint32_t imm_u = insn & 0xfffff000;
    uint32_t imm_j = ((insn >> 21) & 0x3ff) | ((insn >> 10) & 0x400) | (insn & 0x7f800) | ((insn >> 20) & 0x80000);

    
}