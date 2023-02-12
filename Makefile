# Makefile

all: main

main: main.c rns.o rnsv.o
	gcc-10 -g -march=native -Wno-overflow -o main main.c rns.o rnsv.o -lgmp

rns.o: rns.c rns.h structs_data.h 
	gcc-10 -O3 -Wno-overflow -c rns.c -lgmp 
	
rnsv.o: rnsv.c rns.h structs_data.h 
	gcc-10 -O3 -march=native -Wno-overflow -c rnsv.c -lgmp 

clean:
	rm *.o main

