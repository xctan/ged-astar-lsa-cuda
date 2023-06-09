NVCC=/usr/local/cuda/bin/nvcc
NVCCFLAGS=-g -G -std=c++20
OBJS=main.o ged.o heap.o list.o

all: main

build_dir:
	mkdir -p build

%.o: %.cu build_dir
	$(NVCC) $(NVCCFLAGS) -c --device-c $< -o build/$@

main: $(OBJS)
	cd build && $(NVCC) $(NVCCFLAGS) $(OBJS) -o ged

.PHONY: clean

clean:
	rm -rf build