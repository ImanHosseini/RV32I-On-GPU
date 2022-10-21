## Instrumentation
./grv-as main.s (creates a.out?)
riscv64-unknown-elf-objdump -d bpff > odump.txt 2>&1 -mabi=ilp32 -nostdlib -march=rv32im -S ./example/main.c