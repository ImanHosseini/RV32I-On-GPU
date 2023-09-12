## Design
/common/util.cu , util.cuh : stuff that's used everywhere </br> 
/rvcore/rv32.cu : holds the emulator implementation </br>
/apps/* : here are programs using the emulator to do stuff </br>

## BID Instruction
For collecting coverage.
bid zero, 0xb31

## Q1
the constant stuff is defined in rv32.cu (and can't be 'extern'ed: https://forums.developer.nvidia.com/t/constant-memory-variable-with-constant-dont-support-extern-keyword-for-cuda/64831) so we have:
```
// in rv32.cu
__constant__ uint32_t core_mem_size;
```
But we cannot access 'core_mem_size' in another file (e.g. r3.cu), so for example to init that value I defined a function in rv32.cu:
```
void set_cms(int32_t* cms){
    printf("V:%d\n",*cms);
    ccE(cudaMemcpyToSymbol(core_mem_size, cms, sizeof(int32_t)));
}
```
Then extern'ed that function in r3.cu and called it. This approach has at least 1 drawback: say you want to change that function's prototype, now you have to do it twice. </br>
Is there a better way?

## Q2
What is the boundary between rv32.cu (the core emulator implementation) with the stuff that should go in the apps?

## Dynamic Memory
malloc/free/... rely on sbrk() syscall
Option A: Make your own libc: https://wiki.osdev.org/Creating_a_C_Library
Option B: Just implement sbrk and use newlib
Hybrid: A- but just make a slightly tweaked newlib?
Currently "_end" is set in linker script, this is where HEAP starts, newlib uses it

## Binary
SEGMENT:
[cuda_constant]
[cuda_global]
[fuzzing_data]

## Q3
MMU/ Coverage/ etc.

## Fermi-ing
16 GB RAM -> 16Q of 512=<16,32> : 8192 cores, MPC ~ 2MB
64 KB of constant memory
MAP_SIZE = (1  << 10) ~ 1 KB of bitmap