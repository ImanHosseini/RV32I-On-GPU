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
