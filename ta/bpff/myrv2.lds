/* Script for -z combreloc */
/* Copyright (C) 2014-2022 Free Software Foundation, Inc.
   Copying and distribution of this script, with or without modification,
   are permitted in any medium without royalty provided the copyright
   notice and this notice are preserved.  */
/* OUTPUT_FORMAT("elf32-bigriscv") */
OUTPUT_FORMAT("elf32-littleriscv") 
OUTPUT_ARCH(riscv)
ENTRY(_start)
/* SEARCH_DIR("/opt/riscv/riscv64-unknown-elf/lib"); */

/* PHDRS
{
  /* headers PT_PHDR PHDRS ; */
  /* interp PT_INTERP ; */
  /* text PT_LOAD FILEHDR PHDRS ; */
  /* data PT_LOAD ; */
  /* dynamic PT_DYNAMIC ; */
/* }  */

MEMORY
{
    cuda_constant : ORIGIN = 0, LENGTH = 32K
    cuda_global : ORIGIN = 0, LENGTH = 32K
}

SECTIONS
{
    .cuda_constant 0x0 : { *(.text) *(.rodata) *(.fdata)} > cuda_constant
    .cuda_global . : {*(.pkt)} > cuda_global 
}
