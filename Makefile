LIEF_HOME ?= $(HOME)/lief/LIEF-0.13.0-Linux-x86_64/

IMTUI_I=/home/iman/projs/gvz/imtui/include
IMGUI_I=/home/iman/projs/gvz/imtui/third-party/imgui
IMTUI_L=/home/iman/projs/gvz/imtui/build/src
IMGUI_L=/home/iman/projs/gvz/imtui/build/third-party

APPS=./src/apps
ZLIB_DIR=
MARCH=rv32im

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
	nvcc $(APPS)/crvr.cu ./src/rvcore/rv32.cu ./common/util.cu -DDBG -DKDBG0 -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r0

r1: getCCAP
	nvcc -G $(APPS)/r1.cu ./src/rvcore/rv32.cu ./common/util.cu -DDBG -DKDBG0 -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r1

r2: getCCAP
	nvcc -G $(APPS)/r2.cu ./src/rvcore/rv32.cu ./common/util.cu -DDBG -DKDBG0 -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r2

r2_rel: getCCAP
	nvcc $(APPS)/r2.cu ./src/rvcore/rv32.cu ./common/util.cu -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r2_rel

r3: getCCAP
	nvcc $(APPS)/r3.cu ./src/rvcore/rv32.cu ./common/util.cu -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r3

r3dbg: getCCAP
	nvcc -G -DDBG -DKDBG0 $(APPS)/r3.cu ./src/rvcore/rv32.cu ./common/util.cu -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r3dbg

r3cov: getCCAP
	nvcc $(APPS)/r3cov.cu ./src/rvcore/rv32.cu ./common/util.cu -DCOV -I./include -I./common -I$(LIEF_HOME)include -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) -o ./bin/r3cov

grdbg: getCCAP
	nvcc -std=c++17 $(APPS)/grdbg.cu ./src/rvcore/rv32.cu ./common/util.cu -I./include -I./common -I$(LIEF_HOME)include -I$(IMTUI_I) -I$(IMGUI_I) -L$(LIEF_HOME)lib -l:libLIEF.a -arch=$(CCAP) $(IMTUI_L)/libimtui.a $(IMTUI_L)/libimtui-ncurses.a $(IMGUI_L)/libimgui-for-imtui.a -lfmt -lncurses -o ./bin/grdbg

libgrv: ./common/libgrv.c
	riscv64-unknown-elf-gcc -Os -mabi=ilp32 -march=$(RARCH) -nostartfiles -nostdlib -I./common -Wl,-T./ta/grv_ls.lds,--no-relax ./common/libgrv.c -o ./bin/libgrv.a

zlib: 
	$(MAKE) -C ./external/zlib "CC=riscv64-unknown-elf-gcc -mabi=ilp32 -march=$(RARCH) -Wl,-T../../ta/grv_ls.lds"

run-tests: run-t0
build-tests: build-t0

build-t0:
	riscv64-unknown-elf-as -march=$(RARCH) -o ./ta/level0/t1.o ./ta/level0/t1.s
	riscv64-unknown-elf-ld -T./ta/level0/myrv.lds -melf32lriscv -o ./ta/level0/t1 ./ta/level0/t1.o

run-t0:
	./bin/r0