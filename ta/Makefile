.PHONY: bpff_cov bpff_nocov m0
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

bpff_cov: $(ROOT_DIR)/bpff/main.c
	grv-gcc -mabi=ilp32 -Wl,-T$(ROOT_DIR)/grv_ls.lds -nostdlib -march=rv32im -o $(ROOT_DIR)/bin/bpff_cov $(ROOT_DIR)/bpff/main.c

m0: $(ROOT_DIR)/mem/m0.c
	riscv64-unknown-elf-gcc -mabi=ilp32 -march=rv32im -Wl,-T$(ROOT_DIR)/grv_ls.lds -nostartfiles -o $(ROOT_DIR)/bin/m0 $(ROOT_DIR)/mem/m0.c

hello: $(ROOT_DIR)/misc/hello.c
	riscv64-unknown-elf-gcc -mabi=ilp32 -march=rv32im -Wl,-T$(ROOT_DIR)/grv_ls_x.lds -o $(ROOT_DIR)/bin/hello $(ROOT_DIR)/misc/hello.c
	# python3 patch_start.py $(ROOT_DIR)/bin/hello -replace
	riscv64-unknown-elf-objdump -Mno-aliases,numeric -d $(ROOT_DIR)/bin/hello > $(ROOT_DIR)/bin/hello.objdump 2>&1
	readelf -a $(ROOT_DIR)/bin/hello > $(ROOT_DIR)/bin/hello.readelf 2>&1

helloR: $(ROOT_DIR)/misc/hello.c
	riscv64-unknown-elf-gcc -mabi=ilp32 -march=rv32im -o $(ROOT_DIR)/bin/helloR $(ROOT_DIR)/misc/hello.c
	riscv64-unknown-elf-objdump -Mno-aliases,numeric -d $(ROOT_DIR)/bin/helloR > $(ROOT_DIR)/bin/helloR.objdump 2>&1
	readelf -a $(ROOT_DIR)/bin/helloR > $(ROOT_DIR)/bin/helloR.readelf 2>&1