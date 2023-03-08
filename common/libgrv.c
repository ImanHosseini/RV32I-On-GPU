// #define _GNU_SOURCE
// #include <stdio.h>
// #include <dlfcn.h>
#include <stdlib.h>
#include <syscall.h>
// static void* (*real_malloc)(size_t size);
// static void  (*real_free)(void *ptr);

// riscv64-unknown-elf-gcc -mabi=ilp32 -march=rv32im -c -o libgrv.o libgrv.c
// ar rcs libgrv.a libgrv.o

#define LFUNC __attribute__((section (".lib_constant")))

void* malloc(size_t) LFUNC;
void* calloc(size_t, size_t) LFUNC;
void* realloc(void*, size_t) LFUNC;
void free(void*) LFUNC;

// from: https://github.com/riscvarchive/riscv-newlib/blob/riscv-newlib-3.2.0/libgloss/riscv/internal_syscall.h
static inline long
__internal_syscall(long n, long _a0, long _a1, long _a2, long _a3, long _a4, long _a5)
{
  register long a0 asm("a0") = _a0;
  register long a1 asm("a1") = _a1;
  register long a2 asm("a2") = _a2;
  register long a3 asm("a3") = _a3;
  register long a4 asm("a4") = _a4;
  register long a5 asm("a5") = _a5;

#ifdef __riscv_32e
  register long syscall_id asm("t0") = n;
#else
  register long syscall_id asm("a7") = n;
#endif

  asm volatile ("scall"
		: "+r"(a0) : "r"(a1), "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(syscall_id));

  return a0;
}

__attribute__((constructor))
static void init()
{
        return;
}

void *malloc(size_t size)
{
        return __internal_syscall(SYS_malloc, size, 0, 0, 0, 0, 0);
}

void free(void *ptr)
{
        __internal_syscall(SYS_free, ptr, 0, 0, 0, 0, 0);
        return;
}

void *calloc(size_t nitems, size_t size){
    return __internal_syscall(SYS_calloc, size, 0, 0, 0, 0, 0);
}

void *realloc(void* ptr, size_t new_size){
    return __internal_syscall(SYS_realloc, ptr, new_size, 0, 0, 0, 0);
}