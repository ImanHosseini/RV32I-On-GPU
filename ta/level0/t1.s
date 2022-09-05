# riscv64-unknown-elf-as -march=rv32i -o t1.o t1.s
# riscv64-unknown-elf-ld -Tmyrv.lds -melf32lriscv -o t1 t1.o

.global _start      # Provide program starting address to linker

_start: addi  a0, x0, 1      # 1 = StdOut
        auipc x3, 0x12
        # offset is the interpretation of imm by the jal instruction: the contents of imm are shifted left by 1 position and then sign-extended to the size of an address (32 or 64 bits, currently), thus making an integer with a value of -1 million (approximately) to +1 million. YOU PUT TARGET ADDRESS HERE, NOT THE OFFSET ITSELF!
        jal x3, lbl 
        # exit(0)
        ecall    # this will be skipped 
lbl:    
        lui     x4, 0x7
        addi    a0, x0, 0   # Use 0 return code
        addi    a7, x0, 93  # Service command code 93 terminates
        ecall               # Call linux to terminate the program
