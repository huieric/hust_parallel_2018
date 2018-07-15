default: parallel_sudoku

CC = g++

CCFLAGS = -lpthread

SRC = parallel_sudoku.cc

parallel_sudoku: $(SRC)
	$(CC) $(SRC) $(CCFLAGS) -o parallel_sudoku

clean:
	rm -f parallel_sudoku	
