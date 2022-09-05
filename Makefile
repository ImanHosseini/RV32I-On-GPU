LIEF_HOME ?= $(HOME)/lief/LIEF-0.13.0-Linux-x86_64/

# TODO: make this makefile less crappy
# maybe have Makefiles in different dirs? how does that work?

.PHONY: r0 run-tests run-t0 build-tests build-t0

r0:
	nvcc ./src/apps/crvr.cu ./src/rvcore/rv32.cu ./common/util.cu -DDBG -DKDBG0 -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=sm_75 -o ./bin/r0

r1:
	nvcc -g ./src/apps/r1.cu ./src/rvcore/rv32.cu ./common/util.cu -DDBG -DKDBG0 -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=sm_75 -o ./bin/r1

run-tests: run-t0

build-tests: build-t0

build-t0:
	riscv64-unknown-elf-as -march=rv32im -o ./ta/level0/t1.o ./ta/level0/t1.s
	riscv64-unknown-elf-ld -T./ta/level0/myrv.lds -melf32lriscv -o ./ta/level0/t1 ./ta/level0/t1.o

run-t0:
	./bin/r0