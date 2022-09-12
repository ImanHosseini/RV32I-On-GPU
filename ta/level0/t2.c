// from: https://github.com/riscvarchive/riscv-newlib/blob/riscv-newlib-3.2.0/libgloss/riscv/internal_syscall.h
static inline void
_exit(int _a0)
{
  register int a0 asm("a0") = _a0;

#ifdef __riscv_32e
  register long syscall_id asm("t0") = 93;
#else
  register long syscall_id asm("a7") = 93;
#endif

  asm volatile ("scall"
		: "+r"(a0) : "r"(syscall_id));
}

void _start(){
    int x = 12;
    int y = 9;
    int i = 0;
    for(; i < 10; i++){
        int xn = -2*x + 3*y;
        int yn = 3*x  - 2*y;
        x = xn;
        y = yn; 
    }
    _exit(0);
}