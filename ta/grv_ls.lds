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

/*
  The special output section name ‘/DISCARD/’ may be used to discard input sections. Any input sections which are assigned to an output section named ‘/DISCARD/’ are not included in the output file.
  /DISCARD/ :
{
  libc.a ( * )
  libm.a ( * )
  libgcc.a ( * )
}
*/



/* MEMORY
{
    cuda_constant : ORIGIN = 0x0, LENGTH = 64K
    cuda_global : ORIGIN = 0x400000, LENGTH = 0x400000
} */

SECTIONS
{
    /* /DISCARD/ 0 (NOLOAD) : {*(.eh_frame)} */
    __global_pointer$ = .;
    .cuda_constant 0x0 : {*(.lib_constant) *(.cuda_constant) *(.text) *(.text.*) *(.rodata) *(.rodata.*) *(.srodata.*) *(.srodata.*) *(.fdata)} 
    _edata = .; PROVIDE (edata = .);
    .cuda_global . : {*(.cuda_global) *(.pkt) *(.sdata) *(.fini_array) *(.init_array) *(.init_array.*) *(.sdata.*) *(.bss) *(.bss.*) *(.sbss) *(.sbss.*) *(.data.*) *(.eh_frame) *(*)} 
    _end = .; PROVIDE (end = .);
}
