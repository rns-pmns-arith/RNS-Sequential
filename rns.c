#include "rns.h"
#include "rnsv.h"

#include <stdlib.h>

#include <stdio.h>

///////////////////////////////
// RNS equality test
///////////////////////////////
// base : rns base
// pa : A
// pb : B
// return 0 if not equal, 1 if equal
unsigned int rns_equal(struct rns_base_t base, int64_t *pa, int64_t *pb)
{
	unsigned int res = 1;

	for (int i = 0; i < base.size; i++)
	{
		//printf("%lu and %lu\n", pa[i], pb[i]);
		res = res && (pa[i] == pb[i]);
	}
	return res;
}

///////////////////////////////
// RNS addition
///////////////////////////////
// rop : result
// base : RNS base
// pa : A
// pb : B
inline void add_rns(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb)
{
	int j;
	int128 tmp;

	for (j = 0; j < base->size; j++)
	{
		tmp = (int128)pa[j] + pb[j];
		rop[j] = (int64_t)(tmp % base->m[j]);
	}
}

inline void add_rns_cr(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb)
{
	int j;
	int128 tmp;
	
	for (j = 0; j < base->size; j++)
	{
		tmp = (int128)pa[j] + pb[j];
		rop[j] = (int64_t)tmp - (tmp >= base->m[j]) * base->m[j];
	}
}

///////////////////////////////
// RNS substraction
///////////////////////////////
// rop : result
// base : RNS base
// pa : A
// pb : B
inline void sub_rns(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb)
{
	int j;
	int128 tmp;

	for (j = 0; j < base->size; j++)
	{
		tmp = (int128)pa[j] - pb[j];
		rop[j] = (int64_t)(tmp % base->m[j]);
	}
}

inline void sub_rns_cr(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb)
{
	int j;
	int128 tmp;
	int128 up;

	for (j = 0; j < base->size; j++)
	{
		tmp = (int128)pa[j] - pb[j];
		rop[j] = tmp + (tmp < 0) * base->m[j];
	}
}

///////////////////////////////
// RNS multiplication
///////////////////////////////
// rop : result
// base : RNS base
// pa : A
// pb : B
inline void mul_rns(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb)
{
	int j;
	int128 tmp;

	for (j = 0; j < base->size; j++)
	{
		tmp = (int128)pa[j] * pb[j];
		rop[j] = (int64_t)(tmp % base->m[j]);
	}
}

inline void mul_rns_cr(int64_t *rop, struct rns_base_t *base, int64_t *pa, int64_t *pb)
{
	int j;

	for (j = 0; j < base->size; j++)
	{
		rop[j] = mul_mod_cr(pa[j], pb[j], base->k[j]);
	}
}

///////////////////////////////
// RNS Modular multiplication
///////////////////////////////
// rop : result
// pa : A
// pb : B
// mult : constants
// tmp : temporary arrays for intermediate results
void mult_mod_rns(int64_t *rop, int64_t *pa, int64_t *pab, int64_t *pb,
				  int64_t *pbb, struct mod_mul_t *mult, int64_t *tmp[3])
{

	int i;


	mul_rns(tmp[0], mult->conv_atob->rns_a, pa, pb); //A*B
	mul_rns(tmp[1], mult->conv_atob->rns_b, pab, pbb); //A*B in b&se2
	mul_rns(tmp[2], mult->conv_atob->rns_a, tmp[0], mult->inv_p_modMa); //Q*{P-1}
	base_conversion(tmp[0], mult->conv_atob, tmp[2]); // Q in base2
	mul_rns(tmp[2], mult->conv_atob->rns_b, tmp[0], mult->p_modMb); // Q*P base2
	add_rns(tmp[0], mult->conv_atob->rns_b, tmp[1], tmp[2]); // A*B + Q*P in base 2
	mul_rns(rop, mult->conv_atob->rns_b, tmp[0], mult->inv_Ma_modMb); // Division by Ma
}


void mult_mod_rns_cr(int64_t *rop_rnsa, int64_t *rop_rnsb, int64_t *pa, int64_t *pab, int64_t *pb,
					 int64_t *pbb, struct mod_mul_t *mult, int64_t *tmp[4])
{
	
	int i;

	mul_rns_cr(tmp[0], mult->conv_atob->rns_a, pa, pb);					  //A*B
	mul_rns_cr(tmp[1], mult->conv_atob->rns_b, pab, pbb);				  //A*B in base2
	mul_rns_cr(tmp[2], mult->conv_atob->rns_a, tmp[0], mult->inv_p_modMa); //Q*{P-1}
	base_conversion_cr(tmp[0], mult->conv_atob, tmp[2], tmp[3]);			  // Q in base2
	mul_rns_cr(tmp[2], mult->conv_atob->rns_b, tmp[0], mult->p_modMb);	  // Q*P base2
	add_rns_cr(tmp[0], mult->conv_atob->rns_b, tmp[1], tmp[2]);			  // A*B + Q*P in base 2
	mul_rns_cr(rop_rnsb, mult->conv_atob->rns_b, tmp[0], mult->inv_Ma_modMb);	  // Division by Ma rop in base 2
	base_conversion_cr(rop_rnsa, mult->conv_btoa, rop_rnsb, tmp[3]);			  // rop in base 1

}


void mult_mod_rns_cr_cox(int64_t *rop_rnsa, int64_t *rop_rnsb, int64_t *pa, int64_t *pab, int64_t *pb,
						 int64_t *pbb, struct mod_mul_t *mult, int64_t *tmp[4])
{

	int i;
	//int64_t q_hat_mr[1];

	mul_rns_cr(tmp[0], mult->conv_atob->rns_a, pa, pb);					  //A*B
	mul_rns_cr(tmp[1], mult->conv_atob->rns_b, pab, pbb);				  //A*B in base2
	mul_rns_cr(tmp[2], mult->conv_atob->rns_a, tmp[0], mult->inv_p_modMa); //Q*{P-1}
	base_conversion_no_alpha_(tmp[0], mult->conv_atob, tmp[2]);		  //Q hat in base 2
	mul_rns_cr(tmp[2], mult->conv_atob->rns_b, tmp[0], mult->p_modMb);	  // Q*P base2
	add_rns_cr(tmp[0], mult->conv_atob->rns_b, tmp[1], tmp[2]);			  // A*B + Q*P in base 2
	mul_rns_cr(rop_rnsb, mult->conv_atob->rns_b, tmp[0], mult->inv_Ma_modMb);	  // Division by Ma, rop in base 2
	base_conversion_cox(rop_rnsa, mult->conv_btoa, rop_rnsb, 0, 0, 0);	//base_conversion_SK(rop_rnsa, mult->conv_btoa, rop_rnsb, r_hat_mr);

}





///////////////////////////////
// GMP to RNS convertion
///////////////////////////////
//~ Assumes allocation already done for "rop".
void from_int_to_rns(int64_t *rop, struct rns_base_t *base, mpz_t op)
{
	int i;
	mpz_t tmp_residue;
	mpz_init(tmp_residue);
	if (op->_mp_size == 0)
		return;
	for (i = 0; i < base->size; i++)
	{
		rop[i] = mpz_fdiv_r_ui(tmp_residue, op, base->m[i]);  
	}
    
    mpz_clear(tmp_residue);
}

/////////////////////////////////
// Initialize some space for
// the constants of the RNS base
// Computes M
// Computes Mi and inv_Mi
/////////////////////////////////
void init_rns(struct rns_base_t *base)
{
	mpz_t *Mi;
	mpz_t *inv_Mi;
	int i;
	mpz_t tmp_gcd, tmp_mi, tmp, t;
	int64_t *int_inv_Mi;
	int64_t up;

	Mi = (mpz_t *)malloc(base->size * sizeof(mpz_t));
	inv_Mi = (mpz_t *)malloc(base->size * sizeof(mpz_t));
	int_inv_Mi = (int64_t *)malloc(base->size * sizeof(int64_t));
	for (i = 0; i < base->size; i++)
	{
		mpz_init(Mi[i]);
		mpz_init(inv_Mi[i]);
	}
	mpz_init(base->M);
	mpz_init(tmp_gcd);
	mpz_init(tmp_mi);
	mpz_init(tmp);
	mpz_init(t);

	// Computes M
	mpz_add_ui(base->M, base->M, 1);
	for (i = 0; i < base->size; i++)
	{
		mpz_mul_ui(base->M, base->M, base->m[i]);
	}
	// Computes Mi and inv_Mi
	for (i = 0; i < base->size; i++)
	{
		mpz_fdiv_q_ui(Mi[i], base->M, base->m[i]);
		mpz_set_ui(tmp_mi, base->m[i]);
		mpz_gcdext(tmp_gcd, inv_Mi[i], t, Mi[i], tmp_mi);
	}
	base->Mi = Mi;
	base->inv_Mi = inv_Mi;
	// Converts inv_Mi in RNS ie just Inv_Mi mod m_i
	for (i = 0; i < base->size; i++)
	{
		int_inv_Mi[i] = mpz_get_si(inv_Mi[i]);
	}
	base->int_inv_Mi = int_inv_Mi;
	mpz_clears(tmp_gcd, tmp_mi, tmp, t, NULL);
}

/////////////////////////////////
// Clear the space reserved for
// the RNS base constants
/////////////////////////////////
void clear_rns(struct rns_base_t *base)
{
	int i;

	mpz_clear(base->M);
	for (i = 0; i < base->size; i++)
	{
		mpz_clear(base->Mi[i]);
		mpz_clear(base->inv_Mi[i]);
	}
	free(base->Mi);
	free(base->inv_Mi);
}
/////////////////////////////////
// Convert RNS to GMP using
// the CRT
// Assumes "rop" already initialized
/////////////////////////////////
void from_rns_to_int_crt(mpz_t rop, struct rns_base_t *base, int64_t *op)
{
	mpz_t tmp;
	int i;

	// Initializations
	mpz_init(tmp);
	mpz_set_ui(rop, 0);

	for (i = 0; i < base->size; i++)
	{
		mpz_mul_ui(tmp, base->inv_Mi[i], op[i]); //xi*(Mi**(-1))
		mpz_fdiv_r_ui(tmp, tmp, base->m[i]);	 //xi*(Mi**(-1)) mod mi
		mpz_mul(tmp, tmp, base->Mi[i]);
		mpz_add(rop, rop, tmp);
	}
	// Modulo M
	mpz_fdiv_r(rop, rop, base->M);
	mpz_clear(tmp);
}

/////////////////////////////////////////////////
// C function for extended Euclidean Algorithm
// gcd = ax + by
/////////////////////////////////////////////////
int64_t gcdExtended(int64_t a, int64_t b, int64_t *x, int64_t *y)
{
	// Base Case
	if (a == 0)
	{
		*x = 0;
		*y = 1;
		return b;
	}

	int64_t x1, y1; // To store results of recursive call
	int64_t gcd = gcdExtended(b % a, a, &x1, &y1);

	// Update x and y using results of recursive
	// call
	*x = y1 - (b / a) * x1;
	*y = x1;

	return gcd;
}

///////////////////////////////////////////////////////
// Initializes the constant inversese for
// the base conversion using MRS
// inv and mrs are supposed to be already initialized
///////////////////////////////////////////////////////
void initialize_inverses_base_conversion(struct conv_base_t *conv_base)
{
	int i, j;
	int64_t tmp, x;
	int128 tmp2;
	mpz_t tmpz, tmp_residue, tmp_divisor;

	int size = conv_base->rns_a->size;

	mpz_init(tmpz);
	mpz_init(tmp_residue);
	mpz_init(tmp_divisor);

	// Memory allocation for mrs and inverse
	conv_base->inva_to_b = (int64_t **)malloc(size * sizeof(int64_t *));
	conv_base->mrsa_to_b = (int64_t **)malloc(size * sizeof(int64_t *));
	conv_base->Mi_modPi = (int64_t **)malloc(size * sizeof(int64_t *));
	conv_base->invM_modPi = (int64_t *)malloc(size * sizeof(int64_t));

	for (i = 0; i < size; i++)
	{
		conv_base->inva_to_b[i] = (int64_t *)malloc(size * sizeof(int64_t));
		conv_base->mrsa_to_b[i] = (int64_t *)malloc(size * sizeof(int64_t));
		conv_base->Mi_modPi[i] = (int64_t *)malloc(size * sizeof(int64_t));
	}
	// Initialization of the arrays for mrs and inverse
	for (i = 0; i < size; i++)
	{
		for (j = 0; j < size; j++)
		{
			conv_base->inva_to_b[i][j] = 0;
			conv_base->mrsa_to_b[i][j] = 1;
		}
	}
	// Modular inverses : inv[i][j] = inv (m_i) mod m_j
	for (i = 0; i < size - 1; i++)
	{
		for (j = i + 1; j < size; j++)
		{
			gcdExtended(conv_base->rns_a->m[i], conv_base->rns_a->m[j], &x, &tmp);
			if (x > 0) 
				conv_base->inva_to_b[i][j] = x;
			else
				conv_base->inva_to_b[i][j] = x + conv_base->rns_a->m[j];
		}
	}
	// Modular value of the mrs base
	for (j = 0; j < size; j++)
		conv_base->mrsa_to_b[0][j] = conv_base->rns_a->m[0] % conv_base->rns_b->m[j];
	for (i = 1; i < size - 1; i++)
		for (j = 0; j < size; j++)
		{
			tmp2 = (int128)conv_base->mrsa_to_b[i - 1][j] * conv_base->rns_a->m[i];
			conv_base->mrsa_to_b[i][j] = tmp2 % conv_base->rns_b->m[j];
		}
	mpz_neg(tmpz, conv_base->rns_a->M);
	for (i = 0; i < conv_base->rns_b->size; i++)
	{
		mpz_fdiv_r_ui(tmp_residue, tmpz, conv_base->rns_b->m[i]);
		conv_base->invM_modPi[i] = mpz_get_ui(tmp_residue);
	}
	// Mi mod pj
	for (i = 0; i < conv_base->rns_a->size; i++)
	{
		for (j = 0; j < conv_base->rns_b->size; j++)
		{
			mpz_fdiv_r_ui(tmp_residue, conv_base->rns_a->Mi[i], conv_base->rns_b->m[j]);
			conv_base->Mi_modPi[i][j] = mpz_get_ui(tmp_residue);
		}
	}
	mpz_clears(tmpz, tmp_residue, tmp_divisor, NULL);
}

///////////////////////////////////////////////////////
// Converts a RNS number from a base into an other
// using the MRS conversion
///////////////////////////////////////////////////////
void base_conversion(int64_t *rop, struct conv_base_t *conv_base, int64_t *op)
{
	int i, j;
	int64_t a[128]; 
	int64_t tmp;
	int128 tmp2;
	int size = conv_base->rns_a->size;

	for (j = 0; j < size; j++)
		a[j] = op[j];
	for (i = 0; i < size - 1; i++)
	{
		for (j = i + 1; j < size; j++)
		{
			tmp = a[j] - a[i];
			tmp2 = (int128)tmp * conv_base->inva_to_b[i][j];
			a[j] = tmp2 % conv_base->rns_a->m[j];
			if (a[j] < 0) 
				a[j] += conv_base->rns_a->m[j];
		}
	}

	// Residue of the MRS radix
	for (j = 0; j < size; j++)
		rop[j] = a[0] % conv_base->rns_b->m[j];
	for (j = 0; j < size; j++)
	{
		for (i = 1; i < size; i++)
		{
			tmp2 = (int128)a[i] * conv_base->mrsa_to_b[i - 1][j];
			tmp = tmp2 % conv_base->rns_b->m[j];
			rop[j] = ((int128)rop[j] + tmp) % conv_base->rns_b->m[j];

			if (rop[j] < 0) 
				rop[j] += conv_base->rns_b->m[j];
		}
	}
}

///////////////////////////////////////////////////////
// Converts a RNS number from a base into an other
// using the MRS conversion. The RNS base uses Crandall
// numbers
///////////////////////////////////////////////////////
void base_conversion_cr(int64_t *rop, struct conv_base_t *conv_base, int64_t *op, int64_t *a)
{
	int i, j;
	int64_t tmp;
	int128 tmp2;
	int128 tmp3;
	int64_t up, up2, lo, lo2;
	int64_t mask = ((int64_t)1 << 63) - 1; 
	int size = conv_base->rns_a->size;

	for (j = 0; j < size; j++) 
		a[j] = op[j];
	for (i = 0; i < size - 1; i++)
	{
		for (j = i + 1; j < size; j++)
		{
			tmp = a[j] - a[i];
			a[j] = mul_mod_cr_t(tmp, conv_base->inva_to_b[i][j], conv_base->rns_a->k[j]);
		}
	}

	// Residue of the MRS radix
	for (j = 0; j < size; j++)
		rop[j] = (a[0] > conv_base->rns_b->m[j]) ? a[0] - conv_base->rns_b->m[j] : a[0];
	for (j = 0; j < size; j++)
	{
		for (i = 1; i < size; i++)
		{
			tmp = mul_mod_cr(a[i], conv_base->mrsa_to_b[i - 1][j], conv_base->rns_b->k[j]);
			rop[j] = add_mod_cr(rop[j], tmp, conv_base->rns_b->k[j]);
		}
	}
}

//////////////////////////////////////////////
// Cox-rower method for conversion
//////////////////////////////////////////////
int compute_k_cox(int64_t *op, struct rns_base_t *base, int r, int q, int alpha)
{
	int i;
	int n, sigma, k = 0, k_i;
	int64_t mask, mask2, xhi, trunk; 

	r = 63;							 
	q = 7;							 
	alpha = ((int64_t)1 << (q - 1)); 
	mask = ((int64_t)1 << r) - ((int64_t)1 << (r - q));
	mask2 = ((int64_t)1 << q);
	n = base->size;
	sigma = alpha;
	for (i = 0; i < n; i++)
	{
		xhi = mul_mod_cr(op[i], base->int_inv_Mi[i], base->k[i]); // x_i*invM_i mod m_i
		trunk = xhi & mask;
		sigma += trunk >> (r - q);
		k_i = sigma & mask2;
		sigma -= k_i;
		k_i = k_i >> q;
		k += k_i;
	}
	return k;
}

/////////////////////////////////////////////
// Base conversion using Cox-Rower metod
/////////////////////////////////////////////
void base_conversion_cox(int64_t *rop, struct conv_base_t *conv_base, int64_t *op, int r, int q, int alpha)
{
	int i, j;
	int n, k_i;
	int sigma;
	//unsigned char sigma;
	int64_t mask, mask2, xhi, trunk;   
	int size = conv_base->rns_a->size; 
	int64_t tmp, tmp2, tmp3;

	r = 63;							 
	q = 7;							 
	alpha = ((int64_t)1 << (q - 1)); 
	mask = ((int64_t)1 << r) - ((int64_t)1 << (r - q));
	mask2 = ((int64_t)1 << q);
	n = conv_base->rns_a->size;
	sigma = alpha;
	// Initialize rop[]
	for (i = 0; i < n; i++)
		rop[i] = 0;
	for (i = 0; i < n; i++)
	{
		xhi = mul_mod_cr(op[i], conv_base->rns_a->int_inv_Mi[i], conv_base->rns_a->k[i]); // x_i*invM_i mod m_i
		trunk = xhi & mask;
		sigma += trunk >> (r - q);
		k_i = sigma & mask2;
		sigma -= k_i;
		k_i = k_i >> q; // 0 or 1
		for (j = 0; j < size; j++) 
		{
			tmp = mul_mod_cr(xhi, conv_base->Mi_modPi[i][j], conv_base->rns_b->k[j]);
			tmp2 = conv_base->invM_modPi[j] * k_i;
			tmp3 = add_mod_cr(tmp, tmp2, conv_base->rns_b->k[j]);
			rop[j] = add_mod_cr(rop[j], tmp3, conv_base->rns_b->k[j]);
		}
	}
}

void base_conversion_no_alpha_(int64_t *rop, struct conv_base_t *conv_base, int64_t *op)
{

	int i, j;
	int n;
	int size = conv_base->rns_a->size; //Should be the size of secondary base
	int64_t tmp, tmp2, tmp3, sigma;

	n = conv_base->rns_a->size;
	// Initialize rop[]
	for (i = 0; i < n; i++)
		rop[i] = 0;
	for (i = 0; i < n; i++)
	{
		sigma = mul_mod_cr(op[i], conv_base->rns_a->int_inv_Mi[i], conv_base->rns_a->k[i]); // x_i*invM_i mod m_i
		
		
		for (j = 0; j < size; j++) 
		{
			tmp = mul_mod_cr(sigma, conv_base->Mi_modPi[i][j], conv_base->rns_b->k[j]);
			rop[j] = add_mod_cr(rop[j], tmp, conv_base->rns_b->k[j]);
		}
	}
}

///////////////////////////////////////////////////
// Modular addition and multiplication using
// Crandall moduli
///////////////////////////////////////////////////
int64_t add_mod_cr(int64_t a, int64_t b, int k)
{
	int64_t tmp, up, lo, mask = ((int64_t)1 << 63) - 1; 
	int64_t res = 0;									
	uint64_t u_res, u_mod;

	u_mod = ((int64_t)1 << 63) - k; 
	tmp = a + b;
	up = (uint64_t)tmp >> 63; 
	/// Unsigned conversion to avoid sign extension
	lo = tmp & mask;
	res += lo + up * k;
	u_res = (uint64_t)res;
	return res;
}

// Crandal modular multiplication with test
int64_t mul_mod_cr_t(int64_t a, int64_t b, int k)
{
	int128 prod;
	uint128 tmp;
	uint128 tmp2;
	int64_t up;
	int64_t lo;
	int64_t up2, up3;
	int64_t lo2, lo3;
	int64_t mask = ((int64_t)1 << 63) - 1; 
	int64_t res = 0;					   
	uint64_t u_res, u_mod;

	u_mod = ((int64_t)1 << 63) - k; 
	prod = (int128)a * b;
	up = (int64_t)((uint128)prod >> 63); 
	lo = (int64_t)prod & mask;
	tmp = (int128)up * k;
	lo2 = (int64_t)tmp & mask;
	up2 = (int64_t)((uint128)tmp >> 63); 
	tmp2 = (uint128)lo + lo2 + up2 * k;
	up3 = (uint64_t)(tmp2 >> 63); 
	lo3 = (uint64_t)tmp2 & mask;
	res += lo3 + up3 * k;
	u_res = (uint64_t)res;
	if (u_res > u_mod)
	{ 
		res -= u_mod;
	}
	return res;
}

int64_t mul_mod_cr(int64_t a, int64_t b, int k)
{
	int128 prod;
	uint128 tmp;
	uint128 tmp2;
	int64_t up;
	int64_t lo;
	int64_t up2, up3;
	int64_t lo2, lo3;
	int64_t mask = ((int64_t)1 << 63) - 1; 
	int64_t res = 0;					   
	uint64_t u_res, u_mod;

	u_mod = ((int64_t)1 << 63) - k; 
	prod = (int128)a * b;
	up = (int64_t)((uint128)prod >> 63); 
	lo = (int64_t)prod & mask;
	tmp = (int128)up * k;
	lo2 = (int64_t)tmp & mask;
	up2 = (int64_t)((uint128)tmp >> 63); 
	tmp2 = (uint128)lo + lo2 + up2 * k;
	up3 = (uint64_t)(tmp2 >> 63); 
	lo3 = (uint64_t)tmp2 & mask;
	res += lo3 + up3 * k;
	u_res = (uint64_t)res;
	return res;
}
