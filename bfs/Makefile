default: bfs_openmp

CC = g++

CCFLAGES = -openmp

SRC = bfs_openmp.cc

bfs_openmp: $(SRC)
	$(CC) $(SRC) $(CCFLAGES) -o bfs_openmp

clean:
	rm -f bfs_openmp
