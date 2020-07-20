CUDA_HOME ?= /usr/local/cuda
MPI_HOME  ?= /usr/local/openmpi
CC        ?= $(MPI_HOME)/bin/mpicc
CFLAGS    += -I$(CUDA_HOME)/include -I$(MPI_HOME)/include
LDFLAGS   += -L$(CUDA_HOME)/lib64 -L$(MPI_HOME)/lib
LIBS      += -lmpi -lcuda -lcudart

all: ptr_attr.out

%.o: %.c common.h
	$(CC) -c -o $@ $< $(CFLAGS) $(LDFLAGS) $(LIBS)

%.out: %.o common.o
	$(CC) $< common.o -o $@ $(CFLAGS) $(LDFLAGS) $(LIBS)

clean:
	rm -rf *.out *.o

.PHONY: all clean sync
