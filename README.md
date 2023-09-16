# CRV2
## Prereq
### RISCV stuff
https://github.com/riscv-collab/riscv-gnu-toolchain </br>
https://github.com/riscv-software-src/riscv-isa-sim </br>
https://github.com/riscv-software-src/riscv-pk </br>
dont forget:
```
../configure --prefix=$RISCV --host=riscv64-unknown-elf --with-arch=rv32i
```
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
spike --isa=rv32imafc -d /opt/riscv/riscv32-unknown-elf/bin/pk t2
```

## AFL++
```
docker run --rm -it -v $(pwd):"/ta" f29d57145d
```

# Managed0
SYSCALL-level -> can do write. 

# RISCV things
There is one in:
/home/iman/xdr/
But this 1 is in path:
/home/iman/projs/

## DISEMMINATION
[Seventh Workshop on Computer Architecture Research with RISC-V (CARRV 2023)](https://carrv.github.io/2023/)
USENIX ATC? (The home of the legendary QEMU paper)
[VectorVisor - USENIX ATC'23](https://www.usenix.org/conference/atc23/presentation/ginzburg)