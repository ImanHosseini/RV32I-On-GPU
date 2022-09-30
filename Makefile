LIEF_HOME ?= $(HOME)/lief/LIEF-0.13.0-Linux-x86_64/

# TODO: make this makefile less crappy
# maybe have Makefiles in different dirs? how does that work?
# get gpu CC: nvidia-smi --query-gpu=compute_cap --format=csv
# outputs: compute_cap\n8.6
# requires CUDA 11.6+

.PHONY: r0 run-tests run-t0 build-tests build-t0 getCCAP

# `sed` is due to moyix influence :)
getCCAP: 
	$(eval CCAP := sm_$(shell nvidia-smi --query-gpu=compute_cap --format=csv | tail -n 1 | sed 's/\.//'))
	@echo CCAP: $(CCAP)

r0: getCCAP
	nvcc ./src/apps/crvr.cu ./src/rvcore/rv32.cu ./common/util.cu -DDBG -DKDBG0 -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r0

r1: getCCAP
	nvcc -G ./src/apps/r1.cu ./src/rvcore/rv32.cu ./common/util.cu -DDBG -DKDBG0 -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r1

r2: getCCAP
	nvcc -G ./src/apps/r2.cu ./src/rvcore/rv32.cu ./common/util.cu -DDBG -DKDBG0 -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r2

r2_rel: getCCAP
	nvcc ./src/apps/r2.cu ./src/rvcore/rv32.cu ./common/util.cu -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r2_rel

r3: getCCAP
	nvcc ./src/apps/r3.cu ./src/rvcore/rv32.cu ./common/util.cu -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r3

run-tests: run-t0
build-tests: build-t0

build-t0:
	riscv64-unknown-elf-as -march=rv32im -o ./ta/level0/t1.o ./ta/level0/t1.s
	riscv64-unknown-elf-ld -T./ta/level0/myrv.lds -melf32lriscv -o ./ta/level0/t1 ./ta/level0/t1.o

run-t0:
	./bin/r0