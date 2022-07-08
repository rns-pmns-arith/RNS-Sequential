# RNS-Sequential
This repository contains the source code for the RNS sequential and AVX2 operations.

In order to perform the tests, it is necessary to follow the this procedure, in a shell:

- execute the `./rdpmc.sh` script
- go into the directory of the desired version
- type `make -B`
- then `./main`

The Makefile specifies `gcc-10`, which is the gcc version we used in our tests. Previous versions may not be able to compile the source code as it is.

# Parameters
You may want to change the value for the moduli number and/or the modulus $p$.

- In order to change the number of moduli, you have to edit the `structs_data.h` header file, and to change the value of the macro

`#define NB_COEFF 8`

line 21 of the file. It is set to $8$ by default, and you can choose any value less or equal than $64$.

- order to change the modulus $p$, you have to edit the `main.c` file and modify the values in the lines 865, 866 and 867, respectively $p$, $-p^{-1} \mod M$ and $M^{-1} \mod M'$ (see the paper "A software comparison of RNS and PMNS", Algorithm 1 page 2). Please, be careful to ensure that the bound $(n+2)^2\cdot p < M$, as reminded section III.A of the paper, if not, you may have wrong computations.
