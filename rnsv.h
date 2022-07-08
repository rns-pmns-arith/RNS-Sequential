#include "structs_data.h"

#include <immintrin.h>

#include <limits.h>

// ----------------------------------------------------------------------------------------------------------
// Conversions
// -----------

/*__m256i RNS to int64_t RNS conversion using store, more efficient than extract.

BEFORE :
	- base contains the RNS base used to represent op (even if only base->size matters here)
	- op __m256i array to convert

AFTER :
	- rop contains the same values as op

NEEDS :
	- rop allocated

ENSURES :
	- op UNCHANGED
	- base UNCHANGED
*/
void from_m256i_to_int64_t_rns(int64_t *rop, struct rns_base_t *base, __m256i *op);

/*int64_t RNS to __m256i RNS conversion using store, more efficient than extract.

BEFORE :
	- base contains the RNS base used to represent op (even if only base->size matters here)
	- op int64_t array to convert

AFTER :
	- rop contains the same values as op

NEEDS :
	- rop allocated

ENSURES :
	- op UNCHANGED
	- base UNCHANGED
*/
void from_int64_t_to_m256i_rns(__m256i *rop, struct rns_base_t *base, int64_t *op);

// ----------------------------------------------------------------------------------------------------------
// Prints
// ------

/*print the int64_t RNS representation of op in base

BEFORE :
	- base contains the RNS base used to represent op (even if only base->size matters here)
	- op int64_t array to print

ENSURES :
	- op UNCHANGED
	- base UNCHANGED
*/
void print_int64_t_RNS(struct rns_base_t *base, int64_t *op);

/*print values inside of __m256i

BEFORE :
	- op __m256i to print

ENSURES :
	- op UNCHANGED
*/
void print_alone_m256i(__m256i op);

/*print the __m256i RNS representation of op in base

BEFORE :
	- base contains the RNS base used to represent op (even if only base->size matters here)
	- a __m256i array to print

ENSURES :
	- op UNCHANGED
	- base UNCHANGED
*/
void print_m256i_RNS(struct rns_base_t *base, __m256i *op);

// ----------------------------------------------------------------------------------------------------------
// Initializations
// ---------------

/*Initializes avx constants regarding base.

BEFORE :
	- base contains all the correct int64_t constants

NEEDS :
	- base initialized with init_rns

ENSURES :
	- base contains all the needed avx constants to compute operations
*/
void avx_init_rns(struct rns_base_t *base);

/*Initializes avx constants regarding conv_base.

BEFORE :
	- conv_base contains all the correct int64_t constants

NEEDS :
	- conv_base initialized with initialize_inverses_base_conversion

ENSURES :
	- conv_base contains all the needed avx constants to compute operations
*/
void avx_initialize_inverses_base_conversion(struct conv_base_t *conv_base);

/*Initializes avx mrs constants regarding conv_base.

BEFORE :
	- conv_base contains all the correct int64_t and __m256i constants

NEEDS :
	- conv_base initialized with initialize_inverses_base_conversion and avx_initialize_inverses_base_conversion

ENSURES :
	- conv_base contains all the needed avx constants to compute base conversion with mixed radix
*/
void avx_init_mrs(struct conv_base_t *conv_base);

__m256i avx_add_mod_cr(__m256i a, __m256i b, __m256i k);
__m256i avx_sub_mod_cr(__m256i a, __m256i b, __m256i k, __m256i m);

// ----------------------------------------------------------------------------------------------------------
// Addition
// --------

/*__m256i RNS addition.

BEFORE :
	- pa first __m256i array operand
	- pb second __m256i array operand
	- base contains the RNS base used to represent pa and pb

AFTER :
	- rop contains (pa + pb) represented in base 

NEEDS :
	- rop allocated

ENSURES :
	- pa UNCHANGED
	- pb UNCHANGED
	- base UNCHANGED
*/
void avx_add_rns_cr(__m256i *rop, struct rns_base_t *base, __m256i *pa, __m256i *pb);

// ----------------------------------------------------------------------------------------------------------
// Substraction
// ------------

/*__m256i RNS substraction using Crandall moduli RNS base.

BEFORE :
	- pa first __m256i array operand
	- pb second __m256i array operand
	- base contains the RNS base used to represent pa and pb

AFTER :
	- rop contains (pa - pb) represented in base 

NEEDS :
	- rop allocated

ENSURES :
	- pa UNCHANGED
	- pb UNCHANGED
	- base UNCHANGED
*/
void avx_sub_rns_cr(__m256i *rop, struct rns_base_t *base, __m256i *pa, __m256i *pb);

// ----------------------------------------------------------------------------------------------------------
// Multiplication
// --------------

/*__m256i RNS multiplication using Crandall moduli RNS base.

BEFORE :
	- pa first __m256i array operand
	- pb second __m256i array operand
	- base contains the RNS base used to represent pa and pb

AFTER :
	- rop contains (pa * pb) represented in base 

NEEDS :
	- rop allocated

ENSURES :
	- pa UNCHANGED
	- pb UNCHANGED
	- base UNCHANGED
*/
void avx_mul_rns_cr(__m256i *rop, struct rns_base_t *base, __m256i *pa, __m256i *pb);

// ----------------------------------------------------------------------------------------------------------
// Multiplication using Crandall moduli
// -------------------------------------

/*__m256i RNS base conversion using Crandall moduli and mixed-radix representation.

BEFORE :
	- op __m256i in conv_base->rns_a
	- a int64_t RNS representation of op
	- conv_base base conversion constants

AFTER :
	- rop contains op represented in conv_base->rns_b

NEEDS :
	- rop allocated
	- a allocated
	- conv_base correctly initialized

ENSURES :
	- op UNCHANGED
	- conv_base UNCHANGED
*/
void avx_base_conversion_cr(__m256i *rop, struct conv_base_t *conv_base, __m256i *op, int64_t *a);

/*__m256i RNS modular multiplication.

BEFORE :
	- pa first __m256i operand in conv_base->rns_a
	- pab first __m256i operand in conv_base->rns_b
	- pb second __m256i operand in conv_base->rns_a
	- pbb second __m256i operand in conv_base->rns_b
	- mult modular mutliplication constants

AFTER :
	- rop contains (a * b) mod p (where p is in constants of mult)

NEEDS :
	- rop allocated
	- a allocated
	- tmp0, tmp1, tmp2 allocated

ENSURES :
	- pa UNCHANGED
	- pab UNCHANGED
	- pb UNCHANGED
	- pbb UNCHANGED
	- mult UNCHANGED
*/
void avx_mult_mod_rns_cr(__m256i *rop, __m256i *pa, __m256i *pab, __m256i *pb,
						 __m256i *pbb, struct mod_mul_t *mult, __m256i *tmp0, __m256i *tmp1, __m256i *tmp2, int64_t *a);

// ----------------------------------------------------------------------------------------------------------
// Multiplication using Cox-Rower method
// -------------------------------------

/*__m256i RNS base conversion using Cox-Rower method.

BEFORE :
	- op __m256i in conv_base->rns_a
	- conv_base base conversion constants
	- a int64_t RNS representation of op in conv_base->rns_a

AFTER :
	- rop contains op represented in conv_base->rns_b

NEEDS :
	- rop allocated
	- conv_base correctly initialized

ENSURES :
	- op UNCHANGED
	- conv_base UNCHANGED
*/
void avx_base_conversion_cox(__m256i *rop, struct conv_base_t *conv_base, __m256i *op, int64_t *a);

/*__m256i RNS modular multiplication using Cox-Rower method.

BEFORE :
	- pa first __m256i operand in conv_base->rns_a
	- pab first __m256i operand in conv_base->rns_b
	- pb second __m256i operand in conv_base->rns_a
	- pbb second __m256i operand in conv_base->rns_b
	- mult modular mutliplication constants

AFTER :
	- rop contains (a * b) mod p (where p is in constants of mult)

NEEDS :
	- rop allocated
	- a allocated
	- tmp0, tmp1, tmp2 allocated

ENSURES :
	- pa UNCHANGED
	- pab UNCHANGED
	- pb UNCHANGED
	- pbb UNCHANGED
	- mult UNCHANGED
*/
void avx_mult_mod_rns_cr_cox(__m256i *rop, __m256i *pa, __m256i *pab, __m256i *pb,
							 __m256i *pbb, struct mod_mul_t *mult, __m256i *tmp0, __m256i *tmp1, __m256i *tmp2, int64_t *a);
