#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string>

#define FMT_I                  \
    rd = (insn >> 7) & 0x1f;   \
    rs1 = (insn >> 15) & 0x1f; \
    imm = (insn & 0x80000000 ? 0xfffff800 : 0) | ((insn >> 20) & 0x000007ff);

#define FMT_B                  \
    rd = (insn >> 7) & 0x1f;   \
    rs2 = (insn >> 20) & 0x1f; \
    rs1 = (insn >> 15) & 0x1f; \
    imm = (insn & 0x80000000 ? 0xfffff000 : 0) | ((insn << 4) & 0x00000800) | ((insn >> 20) & 0x000007e0) | ((insn >> 7) & 0x0000001e);

#define FMT_J                \
    rd = (insn >> 7) & 0x1f; \
    imm = (insn & 0x80000000 ? 0xfff00000 : 0) | (insn & 0x000ff000) | ((insn & 0x00100000) >> 9) | ((insn & 0x7fe00000) >> 20);

#define FMT_R                  \
    rd = (insn >> 7) & 0x1f;   \
    rs1 = (insn >> 15) & 0x1f; \
    rs2 = (insn >> 20) & 0x1f; \
    rs3 = (insn >> 27) & 0x1f;

#define FMT_S                  \
    rs1 = (insn >> 15) & 0x1f; \
    rs2 = (insn >> 20) & 0x1f; \
    imm = (insn & 0x80000000 ? 0xfffff000 : 0) | ((insn >> 20) & 0xfe0) | ((insn >> 7) & 0x1f);

#define FMT_U                \
    rd = (insn >> 7) & 0x1f; \
    imm = insn & 0xfffff000;

namespace disman
{
    std::string dis1(uint32_t instr, uint32_t pc)
    {
        char buffer[128];
        uint32_t insn = instr;
        uint32_t rd, rs1, rs2, rs3, imm;
        uint32_t insn_masked = instr & 0x7f;
        switch (insn_masked)
        {
        case 0x00000017:
        {
            // RV32I auipc
            FMT_U
            snprintf(buffer, sizeof(buffer), "auipc x%d, 0x%x", rd, imm >> 12);
            return std::string(buffer);
        }
        case 0x0000006f:
        {
            // RV32I jal
            FMT_J
            snprintf(buffer, sizeof(buffer), "jal x%d, 0x%x", rd, pc + imm);
            return std::string(buffer);
        }
        case 0x00000037:
        {
            // RV32I lui
            FMT_U
            snprintf(buffer, sizeof(buffer), "lui x%d, 0x%x", rd, imm >> 12);
            return std::string(buffer);
        }
        case 0x00000027:
        {
            // RV32I bid
            FMT_U
            snprintf(buffer, sizeof(buffer), "bid x%d, 0x%x", rd, imm >> 12);
            return std::string(buffer);
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
                snprintf(buffer, sizeof(buffer), "add x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x00007033:
            {
                // RV32I and
                snprintf(buffer, sizeof(buffer), "and x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x02004033:
            {
                // RV32I div
                snprintf(buffer, sizeof(buffer), "div x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x02005033:
            {
                // RV32I divu
                snprintf(buffer, sizeof(buffer), "divu x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x20000033:
            {
                // RV32I mul
                snprintf(buffer, sizeof(buffer), "mul x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x02001033:
            {
                // RV32M mulh
                snprintf(buffer, sizeof(buffer), "mulh x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x02002033:
            {
                // RV32M mulhsu
                snprintf(buffer, sizeof(buffer), "mulhsu x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x02003033:
            {
                // RV32M mulhu
                snprintf(buffer, sizeof(buffer), "mulhu x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x00006033:
            {
                // RV32I or
                snprintf(buffer, sizeof(buffer), "or x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x02006033:
            {
                // RV32I rem
                snprintf(buffer, sizeof(buffer), "rem x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x02007033:
            {
                // RV32I remu
                snprintf(buffer, sizeof(buffer), "remu x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x00001033:
            {
                // RV32I sll
                snprintf(buffer, sizeof(buffer), "sll x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x00002033:
            {
                // RV32I slt
                snprintf(buffer, sizeof(buffer), "slt x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x00003033:
            {
                // RV32I sltu
                snprintf(buffer, sizeof(buffer), "sltu x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x40005033:
            {
                // RV32I sra
                snprintf(buffer, sizeof(buffer), "sra x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x00005033:
            {
                // RV32I srl
                snprintf(buffer, sizeof(buffer), "srl x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x40000033:
            {
                // RV32I sub
                snprintf(buffer, sizeof(buffer), "sub x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x00004033:
            {
                // RV32I xor
                snprintf(buffer, sizeof(buffer), "xor x%d, x%d, x%d", rd, rs1, rs2);
                return std::string(buffer);
            }
            }
        }
        insn_masked = insn & 0x0000707f;
        switch (insn_masked)
        {
        case 0x00000013:
        {
            // RV32I addi
            FMT_I
            snprintf(buffer, sizeof(buffer), "addi x%d, x%d, %d\n", rd, rs1, (int32_t)imm);
            return std::string(buffer);
        }
        case 0x00007013:
        {
            // RV32I andi
            FMT_I
            snprintf(buffer, sizeof(buffer), "andi x%d, x%d, 0x%x\n", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00000063:
        {
            // RV32I beq
            FMT_B
            snprintf(buffer, sizeof(buffer), "beq x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
            return std::string(buffer);
        }
        case 0x00005063:
        {
            // RV32I bge
            FMT_B
            snprintf(buffer, sizeof(buffer), "bge x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
            return std::string(buffer);
        }
        case 0x00007063:
        {
            // RV32I bgeu
            FMT_B
            snprintf(buffer, sizeof(buffer), "bgeu x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
            return std::string(buffer);
        }
        case 0x00004063:
        {
            // RV32I blt
            FMT_B
            snprintf(buffer, sizeof(buffer), "blt x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
            return std::string(buffer);
        }
        case 0x00006063:
        {
            // RV32I bltu
            FMT_B
            snprintf(buffer, sizeof(buffer), "bltu x%d, x%d, 0x%x\n", rs1, rs2, pc + imm);
            return std::string(buffer);
        }
        case 0x00001063:
        {
            // RV32I bne
            FMT_B
            snprintf(buffer, sizeof(buffer), "bne x%d, x%d, 0x%x", rs1, rs2, pc + imm);
            return std::string(buffer);
        }
        case 0x00000067:
        {
            // RV32I jalr
            FMT_I
            snprintf(buffer, sizeof(buffer), "jalr x%d, x%x, 0x%x\n", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00000003:
        {
            // RV32I lb
            FMT_I
            snprintf(buffer, sizeof(buffer), "lb x%d, x%x, 0x%x\n", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00004003:
        {
            // RV32I lbu
            FMT_I
            snprintf(buffer, sizeof(buffer), "lbu x%d, x%x, 0x%x\n", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00001003:
        {
            // RV32I lh
            FMT_I
            snprintf(buffer, sizeof(buffer), "lh x%d, x%x, 0x%x\n", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00005003:
        {
            // RV32I lhu
            FMT_I
            snprintf(buffer, sizeof(buffer), "lhu x%d, x%x, 0x%x\n", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00002003:
        {
            // RV32I lw
            FMT_I
            snprintf(buffer, sizeof(buffer), "lw x%d, x%x, %d\n", rd, rs1, (int32_t)imm);
            return std::string(buffer);
        }
        case 0x00006013:
        {
            // RV32I ori
            FMT_I
            snprintf(buffer, sizeof(buffer), "ori x%d, x%d, 0x%x", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00004013:
        {
            // RV32I xori
            FMT_I
            snprintf(buffer, sizeof(buffer), "xori x%d, x%d, 0x%x", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00000023:
        {
            FMT_S
            snprintf(buffer, sizeof(buffer), "sb x%d, x%x, 0x%x\n", rs1, rs2, imm);
            return std::string(buffer);
        }
        case 0x00001023:
        {
            // RV32I sh
            FMT_S
            snprintf(buffer, sizeof(buffer), "sh x%d, x%d, 0x%x\n", rs1, rs2, imm);
            return std::string(buffer);
        }
        case 0x00002023:
        {
            // RV32I sw
            FMT_S
            snprintf(buffer, sizeof(buffer), "sw x%d, x%d, %d\n", rs1, rs2, (int32_t)imm);
            return std::string(buffer);
        }
        case 0x00002013:
        {
            // RV32I slti
            FMT_I
            snprintf(buffer, sizeof(buffer), "slti x%d, x%d, 0x%x\n", rd, rs1, imm);
            return std::string(buffer);
        }
        case 0x00003013:
        {
            // RV32I sltiu
            FMT_I
            snprintf(buffer, sizeof(buffer), "sltiu x%d, x%d, 0x%x\n", rd, rs1, imm);
            return std::string(buffer);
        }
        }
        insn_masked = insn & 0xfc00707f;
        {
            FMT_R
            switch (insn_masked)
            {
            case 0x00001013:
            {
                // RV32I slli
                snprintf(buffer, sizeof(buffer), "slli x%d, x%d, x%d\n", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x40005013:
            {
                // RV32I srai
                snprintf(buffer, sizeof(buffer), "srai x%d, x%d, x%d\n", rd, rs1, rs2);
                return std::string(buffer);
            }
            case 0x00005013:
            {
                // RV32I srli
                snprintf(buffer, sizeof(buffer), "srli x%d, x%d, x%d\n", rd, rs1, rs2);
                return std::string(buffer);
            }
            }
        }
        insn_masked = insn & 0xffffffff;
        switch (insn_masked)
        {
        case 0x00100073:
        {
            // RV32I ebreak
            snprintf(buffer, sizeof(buffer), "ebreak\n");
            return std::string(buffer);
        }
        case 0x00000073:
        {
            // RV32I ecall
            snprintf(buffer, sizeof(buffer), "ecall\n");
            return std::string(buffer);
        }
        }
        return std::string("Unknown instruction");
    }
} // namespace disman
