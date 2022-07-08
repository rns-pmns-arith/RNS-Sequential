#include "structs_data.h"

#define _GNU_SOURCE

#include <unistd.h>

#include <stdio.h>

#include <sys/syscall.h>

#define NFUNS 1		 // number of function call between two measures
#define NTEST 500	 // number of test withthe same input data
#define NSAMPLES 10  // number of different input data

inline uint64_t cpucyclesStart(void);
inline uint64_t cpucyclesStop(void);
inline unsigned long rdpmc_instructions(void);