# CRV2
## Prereq
### RISCV stuff
https://github.com/riscv-collab/riscv-gnu-toolchain
https://github.com/riscv-software-src/riscv-isa-sim
https://github.com/riscv-software-src/riscv-pk
I config'ed them with the default prefix of "/opt/riscv" -> then add the bin dir to path.
### LIEF for ELF stuff
Get the lief SDK, notice this line in Makefile:
```
LIEF_HOME ?= $(HOME)/lief/LIEF-0.13.0-Linux-x86_64/
```
## Doing stuff
Look into the Makefile

```
cuda-gdb -q --args ./bin/r1 -np=32 -nq=1 -f=./ta/level0/t1
riscv64-unknown-elf-gcc -mabi=ilp32 -Wl,-Ttext=0x0 -nostdlib -march=rv32im -o t2 t2.c
spike --isa=rv32im pk t2
riscv64-unknown-elf-gcc -mabi=ilp32 -Wl,-Ttext=0x0 -nostdlib -march=rv32im -o t2 t2.c
riscv64-unknown-elf-objdump --disassembler-options=no-aliases -M numeric -d t2 > odump2.txt
```