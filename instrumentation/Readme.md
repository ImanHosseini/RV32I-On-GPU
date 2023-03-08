## Instrumentation
./grv-as main.s (creates a.out?)
riscv64-unknown-elf-objdump -d bpff > odump.txt 2>&1 -mabi=ilp32 -nostdlib -march=rv32im -S ./example/main.c

Do we want to add a '--nocov' flag to skip cov? If you want no cov why are you using grv-gcc? -> Maybe we also want to handle other things with grv-gcc as well.