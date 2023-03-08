PREFIX     ?= /usr/local
BIN_PATH    = $(PREFIX)/bin
GRV_PATH = $(PREFIX)/lib/grv

CFLAGS     ?= -O3 -funroll-loops
CFLAGS     += -Wall -D_FORTIFY_SOURCE=2 -g -Wno-pointer-sign \
		  -DGRV_PATH=\"$(GRV_PATH)\" \
	      -DBIN_PATH=\"$(BIN_PATH)\"

COMM_HDR    = alloc-inl.h config.h debug.h types.h

all: grv-gcc grv-as

grv-gcc: grv-gcc.c $(COMM_HDR)
	$(CC) $(CFLAGS) $@.c -o $@

grv-as: grv-as.c grv-as.h $(COMM_HDR)
	$(CC) $(CFLAGS) $@.c -o $@