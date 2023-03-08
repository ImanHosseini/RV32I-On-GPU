// #define _GNU_SOURCE
// #include <stdio.h>
// #include <dlfcn.h>
#include <stdlib.h>

// static void* (*real_malloc)(size_t size);
// static void  (*real_free)(void *ptr);

// riscv64-unknown-elf-gcc -mabi=ilp32 -march=rv32im -c -o libgrv.o libgrv.c
// ar rcs libgrv.a libgrv.o

__attribute__((constructor))
static void init()
{
        return;
}

void *malloc(size_t size)
{
        return NULL;
}

void free(void *ptr)
{
        return;
}

void *calloc(size_t nitems, size_t size){
    return NULL;
}

void *realloc(void* ptr, size_t new_size){
    return NULL;
}