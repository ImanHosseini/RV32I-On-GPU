.PHONY: bpff_cov bpff_nocov
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

bpff_cov: $(ROOT_DIR)/bpff/main.c
	grv-gcc -mabi=ilp32 -Wl,-T$(ROOT_DIR)/grv_ls.lds -nostdlib -march=rv32im -o $(ROOT_DIR)/bin/bpff_cov $(ROOT_DIR)/bpff/main.c
