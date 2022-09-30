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
SECTIONS
{
    .text 0x0 : { *(.text)}
    .rodata . : {*(.rodata)}
    .fdata . : {*(.fdata)}
    .pkt . : {*(.pkt)}
}
