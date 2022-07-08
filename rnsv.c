#include "rnsv.h"

#include <stdlib.h>

#include <stdio.h>

#include <unistd.h>

#include <stdint.h>

#include <string.h>

#include <time.h>

#include <gmp.h>

#include <immintrin.h>

#include <time.h>

#include "rns.h"

// ----------------------------------------------------------------------------------------------------------
// Conversions
// -----------

inline void from_m256i_to_int64_t_rns(int64_t *rop, struct rns_base_t *base, __m256i *op)
{
	for (int i = 0; i < base->size / 4; i++)
	{
		_mm256_storeu_si256((__m256i *)&rop[4 * i], op[i]);
	}
}

inline void from_int64_t_to_m256i_rns(__m256i *rop, struct rns_base_t *base, int64_t *op)
{
	int j;
	for (j = 0; j < (base->size) / 4; j += 1)
	{
		rop[j] = _mm256_set_epi64x(op[4 * j + 3], op[4 * j + 2], op[4 * j + 1], op[4 * j]);
	}
}

// ----------------------------------------------------------------------------------------------------------
// Prints
// ------

inline void print_int64_t_RNS(struct rns_base_t *base, int64_t *op)
{
	int j;
	for (j = 0; j < base->size; j++)
	{
		printf("%ld ", op[j]);
	}
	printf("\n");
}

inline void print_alone_m256i(__m256i op)
{
	printf("%lld %lld %lld %lld ", _mm256_extract_epi64(op, 3),
		   _mm256_extract_epi64(op, 2),
		   _mm256_extract_epi64(op, 1),
		   _mm256_extract_epi64(op, 0));
}

inline void print_m256i_RNS(struct rns_base_t *base, __m256i *op)
{
	int j;
	for (j = 0; j < (base->size) / 4; j++)
	{
		print_alone_m256i(op[j]);
	}
	printf("\n");
}

// ----------------------------------------------------------------------------------------------------------
// Initializations
// ---------------

inline void avx_init_rns(struct rns_base_t *base)
{

	int n = base->size;

	__m256i *avx_inv_Mi = (__m256i *)_mm_malloc(n * sizeof(__m256i) / 4, 32);

	for (int i = 0; i < n / 4; i++)
	{
		avx_inv_Mi[i] = _mm256_set_epi64x(base->int_inv_Mi[4 * i + 3], base->int_inv_Mi[4 * i + 2], base->int_inv_Mi[4 * i + 1], base->int_inv_Mi[4 * i]);
	}

	base->avx_inv_Mi = avx_inv_Mi;

	__m256i *tmp = (__m256i *)_mm_malloc(n * sizeof(__m256i) / 4, 32);

	for (int i = 0; i < n / 4; i++)
	{
		tmp[i] = _mm256_set_epi64x(base->m[4 * i + 3], base->m[4 * i + 2], base->m[4 * i + 1], base->m[4 * i]);
	}

	base->avx_m = tmp;
}

inline void avx_init_mrs(struct conv_base_t *conv_base)
{
	int i;
	int size = conv_base->rns_a->size;
	__m256i tmp[NB_COEFF / 4];
	conv_base->avx_mrsa_to_b = (__m256i **)malloc(size * sizeof(__m256i *));

	for (i = 0; i < size; i++)
	{
		conv_base->avx_mrsa_to_b[i] = (__m256i *)_mm_malloc(size * sizeof(__m256i) / 4, 32);
	}

	for (i = 0; i < size; i++)
	{
		from_int64_t_to_m256i_rns(conv_base->avx_mrsa_to_b[i], conv_base->rns_a, conv_base->mrsa_to_b[i]);
	}
}

inline void avx_initialize_inverses_base_conversion(struct conv_base_t *conv_base)
{

	int size = conv_base->rns_a->size;

	__m256i **tmp_Arr;

	tmp_Arr = (__m256i **)_mm_malloc(size * sizeof(__m256i *), 32);

	for (int i = 0; i < size; i++)
	{
		tmp_Arr[i] = (__m256i *)_mm_malloc(size * sizeof(__m256i) / 4, 32);
	}

	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size / 4; j++)
		{
			tmp_Arr[i][j] = _mm256_set_epi64x(conv_base->Mi_modPi[i][4 * j + 3], conv_base->Mi_modPi[i][4 * j + 2], conv_base->Mi_modPi[i][4 * j + 1], conv_base->Mi_modPi[i][4 * j]);
		}
	}
	conv_base->avx_Mi_modPi = tmp_Arr;

	conv_base->avx_invM_modPi = (__m256i *)_mm_malloc(size * sizeof(__m256i) / 4, 32);

	for (int i = 0; i < size / 4; i++)
	{
		conv_base->avx_invM_modPi[i] = _mm256_set_epi64x(conv_base->invM_modPi[4 * i + 3], conv_base->invM_modPi[4 * i + 2], conv_base->invM_modPi[4 * i + 1], conv_base->invM_modPi[4 * i]);
	}
}

// ----------------------------------------------------------------------------------------------------------
// Addition
// --------

/* _m256i addition with Crandall moduli.

BEFORE :
	- a first __m256i operand
	- b second __m256i operand
	- k Crandall moduli

AFTER :
	- rop contains (a + b) mod k

NEEDS :
	- rop allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
	- k UNCHANGED
*/
inline __m256i avx_add_mod_cr(__m256i a, __m256i b, __m256i k)
{

	__m256i tmp_mask = _mm256_slli_epi64(_mm256_set1_epi64x(1), 63);
	__m256i mask = _mm256_sub_epi64(tmp_mask, _mm256_set1_epi64x(1));

	__m256i tmp = _mm256_add_epi64(a, b);

	__m256i up = _mm256_srli_epi64(tmp, 63); // La retenue sortante

	__m256i lo = _mm256_and_si256(tmp, mask); // La partie basse de la somme

	__m256i tmp_res = _mm256_madd_epi16(up, k);

	__m256i res = _mm256_add_epi64(lo, tmp_res);

	return res;
}

inline void avx_add_rns_cr(__m256i *rop, struct rns_base_t *base, __m256i *pa, __m256i *pb)
{
	int j;

	for (j = 0; j < (base->size) / 4; j += 1)
	{
		rop[j] = avx_add_mod_cr(pa[j], pb[j], base->avx_k[j]);
	}
}

// ----------------------------------------------------------------------------------------------------------
// Substraction
// ------------

/* _m256i substraction with Crandall moduli.

BEFORE :
	- a first __m256i operand
	- b second __m256i operand
	- k Crandall numbers
	- m Moduli

AFTER :
	- rop contains (a - b) mod k

NEEDS :
	- rop allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
	- k UNCHANGED
	- m UNCHANGED
*/
inline __m256i avx_sub_mod_cr(__m256i a, __m256i b, __m256i k, __m256i m)
{

	__m256i tmp_mask = _mm256_slli_epi64(_mm256_set1_epi64x(1), 63);
	__m256i mask = _mm256_sub_epi64(tmp_mask, _mm256_set1_epi64x(1));

	__m256i tmp1 = _mm256_add_epi64(a, m);

	__m256i tmp = _mm256_sub_epi64(tmp1, b);

	__m256i up = _mm256_srli_epi64(tmp, 63); // La retenue sortante

	__m256i lo = _mm256_and_si256(tmp, mask); // La partie basse de la somme

	__m256i tmp_res = _mm256_madd_epi16(up, k); // mul et add ? Il ne manque pas un terme ?

	__m256i res = _mm256_add_epi64(lo, tmp_res);

	return res;
}

inline void avx_sub_rns_cr(__m256i *rop, struct rns_base_t *base, __m256i *pa, __m256i *pb)
{
	int j;

	for (j = 0; j < (base->size) / 4; j += 1)
	{

		rop[j] = avx_sub_mod_cr(pa[j], pb[j], base->avx_k[j], base->avx_m[j]);
	}
}

// ----------------------------------------------------------------------------------------------------------
// Multiplication
// --------------

static __m256i mask1 = (__m256i){0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL};
static __m256i mask2 = (__m256i){0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL};

/*__m256i addition of 2 terms

BEFORE :
	- a first __m256i operand
	- b second __m256i operand

AFTER :
	- rop_up contains the upper part of (a + b)
	- rop_lo contains the lower part of (a + b)

NEEDS :
	- rop_up allocated
	- rop_lo allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
*/
static inline void avx_add_aux_2e(__m256i *rop_up, __m256i *rop_lo, __m256i a, __m256i b)
{
	__m256i sum = _mm256_add_epi64(a, b);
	*rop_lo = _mm256_and_si256(sum, mask1);
	*rop_up = _mm256_srli_epi64(sum, 63);
}

/*__m256i addition of 3 terms

BEFORE :
	- a first __m256i operand
	- b second __m256i operand
	- c third __m256i operand

AFTER :
	- rop_up contains the upper part of (a + b + c)
	- rop_lo contains the lower part of (a + b + c)

NEEDS :
	- rop_up allocated
	- rop_lo allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
	- c UNCHANGED
*/
static inline void avx_add_aux_3e(__m256i *rop_up, __m256i *rop_lo, __m256i a, __m256i b, __m256i c)
{

	__m256i up, lo, up2, lo2;
	avx_add_aux_2e(&up, &lo, a, b);
	avx_add_aux_2e(&up2, rop_lo, lo, c);
	*rop_up = _mm256_add_epi64(up, up2);
}

static __m256i mask_lo31 = (__m256i){0x7fffffffUL, 0x7fffffffUL, 0x7fffffffUL, 0x7fffffffUL};
static __m256i mask_lo32 = (__m256i){0xffffffffUL, 0xffffffffUL, 0xffffffffUL, 0xffffffffUL};
static __m256i mask_up31 = (__m256i){0x7fffffff00000000UL, 0x7fffffff00000000UL, 0x7fffffff00000000UL, 0x7fffffff00000000UL};
static __m256i mask_up32 = (__m256i){0x7fffffff80000000UL, 0x7fffffff80000000UL, 0x7fffffff80000000UL, 0x7fffffff80000000UL}; //*/

/*__m256i multiplication of 2 terms

BEFORE :
	- a first __m256i operand
	- b second __m256i operand

AFTER :
	- rop_up contains the upper part of (a * b)
	- rop_lo contains the lower part of (a * b)

NEEDS :
	- rop_up allocated
	- rop_lo allocated

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
*/
static inline void avx_mul_aux(__m256i *rop_up, __m256i *rop_lo, __m256i a, __m256i b)
{
	__m256i a_lo = _mm256_and_si256(a, mask_lo31);						  //31 bits
	__m256i a_up = _mm256_srli_epi64(_mm256_and_si256(a, mask_up32), 31); //32 bits
	__m256i b_lo = _mm256_and_si256(b, mask_lo32);						  // 32 bits
	__m256i b_up = _mm256_srli_epi64(_mm256_and_si256(b, mask_up31), 32); //31 bits

	__m256i tmp1 = _mm256_mul_epu32(a_lo, b_lo); //63 bits
	__m256i tmp2 = _mm256_mul_epu32(a_lo, b_up); //62 bits
	__m256i tmp3 = _mm256_mul_epu32(a_up, b_lo); //64 bits
	__m256i tmp4 = _mm256_mul_epu32(a_up, b_up); // 63 bits*/

	__m256i ret;
	avx_add_aux_3e(&ret, rop_lo, tmp1,
				   _mm256_srli_epi64(_mm256_slli_epi64(tmp2, 33), 1),
				   _mm256_srli_epi64(_mm256_slli_epi64(tmp3, 32), 1));

	*rop_up = _mm256_add_epi64(_mm256_srli_epi64(tmp2, 31),
							   _mm256_add_epi64(_mm256_srli_epi64(tmp3, 32),
												_mm256_add_epi64(ret, tmp4)));
}

static __m256i mask = (__m256i){0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL, 0x7fffffffffffffffUL};	  //_mm256_sub_epi64(tmp_mask, _mm256_set1_epi64x(1));
static __m256i tmp_u_mod = (__m256i){0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL, 0x8000000000000000UL}; //_mm256_slli_epi64(_mm256_set1_epi64x(1), 63);

/*__m256i modular multiplication of 2 terms with Crandall moduli

BEFORE :
	- a first __m256i operand
	- b second __m256i operand
	- k Crandall numbers

AFTER :
	- rop contains the upper part of (a * b) mod (n^63-k)

NEEDS :
	- rop

ENSURES :
	- a UNCHANGED
	- b UNCHANGED
	- k UNCHANGED
*/
static inline __m256i avx_mul_mod_cr(__m256i a, __m256i b, __m256i k)
{
	__m256i u_mod = _mm256_sub_epi64(tmp_u_mod, k);
	__m256i up, lo, up2, lo2, up3, lo3;

	avx_mul_aux(&up, &lo, a, b);
	avx_mul_aux(&up2, &lo2, up, k);

	__m256i up2_times_k, ret1, ret2;
	avx_mul_aux(&ret1, &up2_times_k, up2, k);
	avx_add_aux_3e(&ret2, &lo3, lo, lo2, up2_times_k);
	up3 = _mm256_add_epi64(ret1, ret2);
	__m256i res = _mm256_add_epi64(lo3, _mm256_madd_epi16(up3, k));
	return res;
}

inline void avx_mul_rns_cr(__m256i *rop, struct rns_base_t *base, __m256i *pa, __m256i *pb)
{
	int j;

	for (j = 0; j < (base->size) >> 2; j += 1)
	{
		rop[j] = avx_mul_mod_cr(pa[j], pb[j], base->avx_k[j]);
	}
}

// ----------------------------------------------------------------------------------------------------------
// Multiplication using Crandall moduli
// -------------------------------------

inline void avx_base_conversion_cr(__m256i *rop, struct conv_base_t *conv_base, __m256i *op, int64_t *a)
{
	int i, j;
	__m256i avx_tmp;
	int64_t tmp;
	int size = conv_base->rns_a->size;

	for (i = 0; i < size - 1; i++)
	{
		for (j = i + 1; j < size; j++)
		{
			tmp = a[j] - a[i];
			a[j] = mul_mod_cr(tmp, conv_base->inva_to_b[i][j], conv_base->rns_a->k[j]);
		}
	}

	__m256i a0_256 = _mm256_set1_epi64x(a[0]);

	for (j = 0; j < size / 4; j++)
	{
		__m256i b256 = _mm256_lddqu_si256((__m256i *)&conv_base->rns_b->m[j >> 2]);
		__m256i cmp256 = _mm256_cmpgt_epi64(a0_256, b256);
		rop[j] = _mm256_sub_epi64(a0_256, _mm256_and_si256(cmp256, b256));
	}

	for (j = 0; j < size / 4; j++)
	{
		for (i = 1; i < size; i++)
		{
			avx_tmp = avx_mul_mod_cr(_mm256_set1_epi64x(a[i]), conv_base->avx_mrsa_to_b[i - 1][j], conv_base->rns_b->avx_k[j]);
			rop[j] = avx_add_mod_cr(rop[j], avx_tmp, conv_base->rns_b->avx_k[j]);
		}
	}
}

inline void avx_mult_mod_rns_cr(__m256i *rop, __m256i *pa, __m256i *pab, __m256i *pb,
								__m256i *pbb, struct mod_mul_t *mult, __m256i *tmp0, __m256i *tmp1, __m256i *tmp2, int64_t *a)
{

	avx_mul_rns_cr(tmp0, mult->conv_atob->rns_a, pa, pb);					  //A*B
	avx_mul_rns_cr(tmp1, mult->conv_atob->rns_b, pab, pbb);					  //A*B in base2
	avx_mul_rns_cr(tmp2, mult->conv_atob->rns_a, tmp0, mult->avx_inv_p_modMa); //Q*{P-1}
	from_m256i_to_int64_t_rns(a, mult->conv_atob->rns_a, tmp2);				  //storing tmp2 in a
	avx_base_conversion_cr(tmp0, mult->conv_atob, tmp2, a);					  // Q in base2
	avx_mul_rns_cr(tmp2, mult->conv_atob->rns_b, tmp0, mult->avx_p_modMb);	  // Q*P base2
	avx_add_rns_cr(tmp0, mult->conv_atob->rns_b, tmp1, tmp2);				  // A*B + Q*P in base 2
	avx_mul_rns_cr(rop, mult->conv_atob->rns_b, tmp0, mult->avx_inv_Ma_modMb); // Division by Ma
}

// ----------------------------------------------------------------------------------------------------------
// Multiplication using Cox-Rower method
// -------------------------------------

static __m256i cox_mask = (__m256i){0x7F00000000000000UL, 0x7F00000000000000UL, 0x7F00000000000000UL, 0x7F00000000000000UL};
static __m256i cox_mask2 = (__m256i){0x80UL, 0x80UL, 0x80UL, 0x80UL};
static __m256i cox_sigma = (__m256i){0x40UL, 0x40UL, 0x40UL, 0x40UL};

inline void avx_base_conversion_cox(__m256i *rop, struct conv_base_t *conv_base, __m256i *op, int64_t *a)
{
	int i, j;
	int size = conv_base->rns_a->size;

	int r = 63;
	int q = 7;

	__m256i sigma = cox_sigma;

	for (i = 0; i < size / 4; i++)
	{
		rop[i] = _mm256_set1_epi64x(0);
	}

	__m256i xhi, trunk, k_i;
	__m256i tmp0, tmp1, tmp2;

	for (i = 0; i < size; i++)
	{
		xhi = _mm256_set1_epi64x(mul_mod_cr(a[i], conv_base->rns_a->int_inv_Mi[i], conv_base->rns_a->k[i]));

		trunk = xhi & cox_mask;
		sigma = _mm256_add_epi64(sigma, _mm256_srli_epi64(trunk, r - q));
		k_i = sigma & cox_mask2;

		sigma = _mm256_sub_epi64(sigma, k_i);
		k_i = _mm256_srli_epi64(k_i, q);

		for (j = 0; j < size / 4; j++)
		{
			tmp0 = avx_mul_mod_cr(xhi, conv_base->avx_Mi_modPi[i][j], conv_base->rns_b->avx_k[j]);
			tmp1 = conv_base->avx_invM_modPi[j] * k_i;
			tmp2 = avx_add_mod_cr(tmp0, tmp1, conv_base->rns_b->avx_k[j]);
			rop[j] = avx_add_mod_cr(rop[j], tmp2, conv_base->rns_b->avx_k[j]);
		}
	}
}

inline void avx_mult_mod_rns_cr_cox(__m256i *rop, __m256i *pa, __m256i *pab, __m256i *pb,
									__m256i *pbb, struct mod_mul_t *mult, __m256i *tmp0, __m256i *tmp1, __m256i *tmp2, int64_t *a)
{

	int i;

	avx_mul_rns_cr(tmp0, mult->conv_atob->rns_a, pa, pb);					  //A*B
	avx_mul_rns_cr(tmp1, mult->conv_atob->rns_b, pab, pbb);					  //A*B in base2
	avx_mul_rns_cr(tmp2, mult->conv_atob->rns_a, tmp0, mult->avx_inv_p_modMa); //Q*{P-1}
	from_m256i_to_int64_t_rns(a, mult->conv_atob->rns_a, tmp2);				  //storing tmp2 in a
	avx_base_conversion_cox(tmp0, mult->conv_atob, tmp2, a);					  //Q in base 2
	avx_mul_rns_cr(tmp2, mult->conv_atob->rns_b, tmp0, mult->avx_p_modMb);	  // Q*P base2
	avx_add_rns_cr(tmp0, mult->conv_atob->rns_b, tmp1, tmp2);				  // A*B + Q*P in base 2
	avx_mul_rns_cr(rop, mult->conv_atob->rns_b, tmp0, mult->avx_inv_Ma_modMb); // Division by Ma
}
