#include <stdlib.h>

#include <stdio.h>

#include <gmp.h>

#include <math.h>

#include "rns.h"

#include "tests.c"

#include "rnsv.h"

#include "base64.c"

int main(void)
{

	FILE *fpt;

	fpt = fopen("results/Results.json", "w+");

	fprintf(fpt, "{\n");

	printf("\nVectorized RNS timing :\n");

	// Initializing random
	gmp_randstate_t state;
	gmp_randinit_default(state);
	// Timers
	unsigned long long timer, t1, t2;
	// Variables
	int64_t op1[NB_COEFF];
	int64_t op2[NB_COEFF];
	int64_t res[NB_COEFF];
	
	__m256i avx_op1[NB_COEFF / 4];
	__m256i avx_op2[NB_COEFF / 4];
	__m256i avx_res[NB_COEFF / 4];

	mpz_t A, B, C;
	mpz_inits(A, B, C, NULL);

	// Base
	struct rns_base_t rns_a;
	rns_a.size = NB_COEFF;

    int64_t * m_tmp = m1;
	rns_a.m = m_tmp;

    int * k_tmp = k1;
	rns_a.k = k_tmp;
 	init_rns(&rns_a);
	avx_init_rns(&rns_a);

	int64_t tmp_k[NB_COEFF];
	int j;
	for (j = 0; j < NB_COEFF; j++)
	{
		tmp_k[j] = (int64_t)k_tmp[j];
	}

	__m256i avx_k[NB_COEFF / 4];
	from_int64_t_to_m256i_rns(avx_k, &rns_a, tmp_k);
	rns_a.avx_k = avx_k;

	mpz_t M;
	mpz_inits(M, NULL);
	mpz_set(M, rns_a.M); // Get M from the base
	unsigned long long int timing = ULLONG_MAX;
	unsigned long before_cycles, after_cycles, cycles = ULONG_MAX;
	unsigned long before_instructions, after_instructions, instructions = ULONG_MAX;
	unsigned long before_ref, after_ref, ref = ULONG_MAX;


	printf("\n\tBase :\n\nNB_COEFF = %d\n",NB_COEFF);

	for (int i = 0; i < NB_COEFF; i++)
	{
		printf("\t\t %ld\n", m_tmp[i]);
	}

	printf("\n1. Multiplication :\n");

	fprintf(fpt, "\"multiplication\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		mul_rns_cr(res, &rns_a, op1, op2);
	}

	printf("Done.\n");

	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			mul_rns_cr(res, &rns_a, op1, op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			mul_rns_cr(res, &rns_a, op1, op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			mul_rns_cr(res, &rns_a, op1, op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			mul_rns_cr(res, &rns_a, op1, op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}
	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential multiplication : %ld instructions.\n", instructions);
	printf("\tRNS sequential multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential multiplication : %ld reference CPU cycles.\n", ref);

	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	//printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); //Randomly generates A < M
	mpz_urandomm(B, state, M); //Randomly generates B < M

	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);
	from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);
	from_int64_t_to_m256i_rns(avx_op2, &rns_a, op2);

	for (int i = 0; i < NTEST; i++)
	{

		avx_mul_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);
	}

	printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);
		from_int64_t_to_m256i_rns(avx_op2, &rns_a, op2);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx_mul_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx_mul_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx_mul_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx_mul_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS vectorized multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS vectorized multiplication : %ld instructions.\n", instructions);
	printf("\tRNS vectorized multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS vectorized multiplication : %ld reference CPU cycles.\n", ref);

	printf("\n\n2. Addition :\n");
	fprintf(fpt, "\"addition\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	//printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		add_rns_cr(res, &rns_a, op1, op2);
	}
	//printf("Done.\n");

	//printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			add_rns_cr(res, &rns_a, op1, op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			add_rns_cr(res, &rns_a, op1, op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			add_rns_cr(res, &rns_a, op1, op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			add_rns_cr(res, &rns_a, op1, op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential addition : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential addition : %ld instructions.\n", instructions);
	printf("\tRNS sequential addition : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential addition : %ld reference CPU cycles.\n", ref);

	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); //Randomly generates A < M
	mpz_urandomm(B, state, M); //Randomly generates B < M

	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);
	from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);
	from_int64_t_to_m256i_rns(avx_op2, &rns_a, op2);

	for (int i = 0; i < NTEST; i++)
	{

		avx_add_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);
	}

	printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);
		from_int64_t_to_m256i_rns(avx_op2, &rns_a, op2);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx_add_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx_add_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx_add_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx_add_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_ref = rdpmc_reference_cycles();
			//printf("%ld\n", after_ref-before_ref);

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS vectorized addition : %lld CPU cycles.\n", timing);
	printf("\tRNS vectorized addition : %ld instructions.\n", instructions);
	printf("\tRNS vectorized addition : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS vectorized addition : %ld reference CPU cycles.\n", ref);

	printf("\n\n3. Substraction :\n");

	fprintf(fpt, "\"substraction\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		sub_rns_cr(res, &rns_a, op1, op2);
	}
	printf("Done.\n");

	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			sub_rns_cr(res, &rns_a, op1, op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			sub_rns_cr(res, &rns_a, op1, op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			sub_rns_cr(res, &rns_a, op1, op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			sub_rns_cr(res, &rns_a, op1, op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential substraction : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential substraction : %ld instructions.\n", instructions);
	printf("\tRNS sequential substraction : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential substraction : %ld reference CPU cycles.\n", ref);

	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); //Randomly generates A < M
	mpz_urandomm(B, state, M); //Randomly generates B < M

	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);
	from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);
	from_int64_t_to_m256i_rns(avx_op2, &rns_a, op2);

	for (int i = 0; i < NTEST; i++)
	{

		avx_sub_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);
	}

	printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		from_int_to_rns(op2, &rns_a, B);
		from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);
		from_int64_t_to_m256i_rns(avx_op2, &rns_a, op2);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx_sub_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx_sub_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx_sub_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx_sub_rns_cr(avx_res, &rns_a, avx_op1, avx_op2);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS vectorized substraction : %lld CPU cycles.\n", timing);
	printf("\tRNS vectorized substraction : %ld instructions.\n", instructions);
	printf("\tRNS vectorized substraction : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS vectorized substraction : %ld reference CPU cycles.\n", ref);

	// Second Base
	struct rns_base_t rns_b;

    int64_t * base2_bis = m2;

    int * k2_bis = k2;
	rns_b.size = NB_COEFF;
	rns_b.m = base2_bis;
	rns_b.k = k2_bis;
	init_rns(&rns_b);

	int64_t tmp_k2[NB_COEFF];
	for (j = 0; j < NB_COEFF; j++)
	{
		tmp_k2[j] = (int64_t)k2_bis[j];
	}
	__m256i avx_k2[NB_COEFF / 4];
	from_int64_t_to_m256i_rns(avx_k2, &rns_b, tmp_k2);
	rns_b.avx_k = avx_k2;

	// Base conversion
	struct conv_base_t conv_atob;
	conv_atob.rns_a = &rns_a;
	conv_atob.rns_b = &rns_b;
	initialize_inverses_base_conversion(&conv_atob);
	struct conv_base_t conv_btoa;
	conv_btoa.rns_a = &rns_b;
	conv_btoa.rns_b = &rns_a;
	initialize_inverses_base_conversion(&conv_btoa);

	int64_t ttt[NB_COEFF];

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	printf("\n\n4. Base conversion :\n");

	fprintf(fpt, "\"base_conversion\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		base_conversion_cr(op2, &conv_atob, op1, ttt);
	}
	printf("Done.\n");

	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			base_conversion_cr(op2, &conv_atob, op1, ttt);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			base_conversion_cr(op2, &conv_atob, op1, ttt);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			base_conversion_cr(op2, &conv_atob, op1, ttt);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			base_conversion_cr(op2, &conv_atob, op1, ttt);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential base conversion : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential base conversion : %ld instructions.\n", instructions);
	printf("\tRNS sequential base conversion : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential base conversion : %ld reference CPU cycles.\n", ref);

	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); //Randomly generates A < M

	avx_init_mrs(&conv_atob);
	avx_initialize_inverses_base_conversion(&conv_atob);

	from_int_to_rns(op1, &rns_a, A);
	from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);

	for (int i = 0; i < NTEST; i++)
	{

		avx_base_conversion_cr(avx_op2, &conv_atob, avx_op1, op1);
	}

	printf("Done.\n");

	printf("\tTesting... ");

	// timing
	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		from_int_to_rns(op1, &rns_a, A);
		from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);
		for (int j = 0; j < NTEST; j++)
		{
			// RDTSC
			t1 = cpucyclesStart();

			avx_base_conversion_cr(avx_op2, &conv_atob, avx_op1, op1);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx_base_conversion_cr(avx_op2, &conv_atob, avx_op1, op1);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx_base_conversion_cr(avx_op2, &conv_atob, avx_op1, op1);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx_base_conversion_cr(avx_op2, &conv_atob, avx_op1, op1);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS vectorized base conversion : %lld CPU cycles.\n", timing);
	printf("\tRNS vectorized base conversion : %ld instructions.\n", instructions);
	printf("\tRNS vectorized base conversion : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS vectorized base conversion : %ld reference CPU cycles.\n", ref);

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	//getchar();

	// MODULAR MULTIPLICATION
	printf("\n\n5. Modular multiplication :\n");

	fprintf(fpt, "\"modular_multiplication\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	mpz_t inv_p_modM, inv_M_modMp, modul_p;
	mpz_inits(inv_p_modM, inv_M_modMp, modul_p, NULL);

	int64_t pa[NB_COEFF];
	int64_t pb[NB_COEFF];
	int64_t pab[NB_COEFF];
	int64_t pbb[NB_COEFF];
	int64_t pc[NB_COEFF];
	int64_t pca[NB_COEFF];
	int64_t pcb[NB_COEFF];
	int64_t pp1[NB_COEFF];
	int64_t pp2[NB_COEFF];
	int64_t pp3[NB_COEFF];

	// Set custom values
	mpz_set_str(modul_p, "115792089021636622262124715160334756877804245386980633020041035952359812890593", 10);
	mpz_set_str(inv_p_modM, "-7210642370083763919688086698199040857322895088554003933210287226647459666846134833419938084604981461493089686639677942359747717700454441525223348684285", 10);
	mpz_set_str(inv_M_modMp, "2926906825829426928727294150364906856635623568440932569450673109926460590684432927230290255276608760237299661987870702836538185953568700154975953006659", 10);

	// Initialization
	base_conversion_cr(pb, &conv_atob, pa, ttt);

	//Modular multiplication

	struct mod_mul_t mult;
	mpz_t tmp_gcd, t, tmp_inv;

	mpz_init(tmp_gcd);
	mpz_init(t);
	mpz_init(tmp_inv);
	from_int_to_rns(pp2, &rns_b, modul_p); // P mod Mb

	mpz_sub(tmp_inv, rns_a.M, modul_p);
	mpz_gcdext(tmp_gcd, inv_p_modM, t, tmp_inv, rns_a.M);
	from_int_to_rns(pp1, &rns_a, inv_p_modM); //(-P)^-1 mod Ma

	mpz_gcdext(tmp_gcd, inv_M_modMp, t, rns_a.M, rns_b.M);
	from_int_to_rns(pp3, &rns_b, inv_M_modMp); // Ma^{-1} mod Mb

	mult.inv_p_modMa = pp1;
	mult.p_modMb = pp2;
	mult.inv_Ma_modMb = pp3;
	mult.conv_atob = &conv_atob;
	mult.conv_btoa = &conv_btoa;

	int64_t *tmp[4]; // RNS modular multiplication intermediate results
	// One more for the base convertion
	tmp[0] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[1] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[2] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));
	tmp[3] = (int64_t *)malloc(NB_COEFF * sizeof(int64_t));

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P
	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	// Heating caches
	printf("\n\tHeating caches... ");
	/*for (int i = 0; i < NTEST; i++)
	{
		mult_mod_rns_cr(pca,pcb, pa, pab, pb, pbb, &mult, tmp);
		getchar();

	}//*/
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, modul_p); //Randomly generates A < M
		mpz_urandomm(B, state, modul_p); //Randomly generates B < M
		from_int_to_rns(pa, &rns_a, A);
		from_int_to_rns(pb, &rns_a, B);
		from_int_to_rns(pab, &rns_b, A);
		from_int_to_rns(pbb, &rns_b, B);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			mult_mod_rns_cr(pca,pcb, pa, pab, pb, pbb, &mult, tmp);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			mult_mod_rns_cr(pca,pcb, pa, pab, pb, pbb, &mult, tmp);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			mult_mod_rns_cr(pca,pcb, pa, pab, pb, pbb, &mult, tmp);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			mult_mod_rns_cr(pca,pcb, pa, pab, pb, pbb, &mult, tmp);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential modular multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential modular multiplication : %ld instructions.\n", instructions);
	printf("\tRNS sequential modular multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential modular multiplication : %ld reference CPU cycles.\n", ref);//

	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	__m256i avx_pa[NB_COEFF / 4];
	__m256i avx_pb[NB_COEFF / 4];
	__m256i avx_pab[NB_COEFF / 4];
	__m256i avx_pbb[NB_COEFF / 4];

	__m256i avx_pp1[NB_COEFF / 4];
	__m256i avx_pp2[NB_COEFF / 4];
	__m256i avx_pp3[NB_COEFF / 4];

	from_int64_t_to_m256i_rns(avx_pp1, &rns_a, pp1);
	from_int64_t_to_m256i_rns(avx_pp2, &rns_b, pp2);
	from_int64_t_to_m256i_rns(avx_pp3, &rns_b, pp3);

	mult.avx_inv_p_modMa = avx_pp1;
	mult.avx_p_modMb = avx_pp2;
	mult.avx_inv_Ma_modMb = avx_pp3;

	int64_t *a = malloc(sizeof(int64_t));
	__m256i tmp0[NB_COEFF / 4];
	__m256i tmp1[NB_COEFF / 4];
	__m256i tmp2[NB_COEFF / 4];
	// Using an array is less efficient

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P
	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	from_int64_t_to_m256i_rns(avx_pa, &rns_a, pa);
	from_int64_t_to_m256i_rns(avx_pb, &rns_a, pb);
	from_int64_t_to_m256i_rns(avx_pab, &rns_b, pab);
	from_int64_t_to_m256i_rns(avx_pbb, &rns_b, pbb);

	// Heating caches
	printf("\n\tHeating caches... ");
	for (int i = 0; i < NTEST; i++)
	{

		avx_mult_mod_rns_cr(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);
	}
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, modul_p); //Randomly generates A < P
		mpz_urandomm(B, state, modul_p); //Randomly generates A < P

		from_int_to_rns(pa, &rns_a, A);
		from_int_to_rns(pb, &rns_a, B);
		from_int_to_rns(pab, &rns_b, A);
		from_int_to_rns(pbb, &rns_b, B);

		from_int64_t_to_m256i_rns(avx_pa, &rns_a, pa);
		from_int64_t_to_m256i_rns(avx_pb, &rns_a, pb);
		from_int64_t_to_m256i_rns(avx_pab, &rns_b, pab);
		from_int64_t_to_m256i_rns(avx_pbb, &rns_b, pbb);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx_mult_mod_rns_cr(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx_mult_mod_rns_cr(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx_mult_mod_rns_cr(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx_mult_mod_rns_cr(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS parallel modular multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS parallel modular multiplication : %ld instructions.\n", instructions);
	printf("\tRNS parallel modular multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS parallel modular multiplication : %ld reference CPU cycles.\n", ref);

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	free(a);

	// Cox conversion
	printf("\n\n7. Cox base conversion\n");

	fprintf(fpt, "\"cox_base_conv\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	printf("\n\tHeating caches... ");
	mpz_urandomm(A, state, M); // Randomly generates A < M
	mpz_urandomm(B, state, M); // Randomly generated B < M
	from_int_to_rns(op1, &rns_a, A);
	from_int_to_rns(op2, &rns_a, B);

	for (int i = 0; i < NTEST; i++)
	{
		base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);
	}
	printf("Done.\n");

	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{
		mpz_urandomm(A, state, M); //Randomly generates A < M
		mpz_urandomm(B, state, M); //Randomly generates B < M
		from_int_to_rns(op1, &rns_a, A);
		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			base_conversion_cox(op2, &conv_atob, op1, 0, 0, 0);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}

		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential cox base conversion : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential cox base conversion : %ld instructions.\n", instructions);
	printf("\tRNS sequential cox base conversion : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential cox base conversion : %ld reference CPU cycles.\n", ref);

	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	// Vectored constants needed
	avx_initialize_inverses_base_conversion(&conv_atob);

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P
	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	// Heating caches
	printf("\n\tHeating caches... ");

	mpz_urandomm(A, state, M);
	from_int_to_rns(op1, &rns_a, A);
	from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);
	for (int i = 0; i < NTEST; i++)
	{

		avx_base_conversion_cox(avx_res, &conv_atob, avx_op1, op1);
	}
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{

		mpz_urandomm(A, state, modul_p); //Randomly generates A < P
		from_int_to_rns(op1, &rns_a, A);
		from_int64_t_to_m256i_rns(avx_op1, &rns_a, op1);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx_base_conversion_cox(avx_res, &conv_atob, avx_op1, op1);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx_base_conversion_cox(avx_res, &conv_atob, avx_op1, op1);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx_base_conversion_cox(avx_res, &conv_atob, avx_op1, op1);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx_base_conversion_cox(avx_res, &conv_atob, avx_op1, op1);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t},\n");

	printf("Done.\n");
	printf("\tRNS parallel cox base conversion : %lld CPU cycles.\n", timing);
	printf("\tRNS parallel cox base conversion : %ld instructions.\n", instructions);
	printf("\tRNS parallel cox base conversion : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS parallel cox base conversion : %ld reference CPU cycles.\n", ref);

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	// Cox modular multiplication
	printf("\n\n6. Cox modular multiplication :\n");

	fprintf(fpt, "\"cox_mod_mul\" :\n\t{\n");
	fprintf(fpt, "\t\"sequential\" :\n\t\t[\n");

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P
	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	// Heating caches
	printf("\n\tHeating caches... ");

	mpz_urandomm(A, state, M);
	from_int_to_rns(op1, &rns_a, A);
	for (int i = 0; i < NTEST; i++)
	{

		mult_mod_rns_cr_cox(pca,pcb, pa, pab, pb, pbb, &mult, tmp);
	}
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{

		mpz_urandomm(A, state, modul_p); //Randomly generates A < P
		mpz_urandomm(B, state, modul_p); //Randomly generates A < P
		from_int_to_rns(pa, &rns_a, A);
		from_int_to_rns(pb, &rns_a, B);
		from_int_to_rns(pab, &rns_b, A);
		from_int_to_rns(pbb, &rns_b, B);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			mult_mod_rns_cr_cox(pca,pcb, pa, pab, pb, pbb, &mult, tmp);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			mult_mod_rns_cr_cox(pca,pcb, pa, pab, pb, pbb, &mult, tmp);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			mult_mod_rns_cr_cox(pca,pcb, pa, pab, pb, pbb, &mult, tmp);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			mult_mod_rns_cr_cox(pca,pcb, pa, pab, pb, pbb, &mult, tmp);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t],\n");

	printf("Done.\n");
	printf("\tRNS sequential cox modular multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS sequential cox modular multiplication : %ld instructions.\n", instructions);
	printf("\tRNS sequential cox modular multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS sequential cox modular multiplication : %ld reference CPU cycles.\n", ref);

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	fprintf(fpt, "\t\"parallel\" :\n\t\t[\n");

	// Heating caches
	printf("\n\tHeating caches... ");

	mpz_urandomm(A, state, modul_p); //Randomly generates A < P
	mpz_urandomm(B, state, modul_p); //Randomly generates A < P

	from_int_to_rns(pa, &rns_a, A);
	from_int_to_rns(pb, &rns_a, B);
	from_int_to_rns(pab, &rns_b, A);
	from_int_to_rns(pbb, &rns_b, B);

	from_int64_t_to_m256i_rns(avx_pa, &rns_a, pa);
	from_int64_t_to_m256i_rns(avx_pb, &rns_a, pb);
	from_int64_t_to_m256i_rns(avx_pab, &rns_b, pab);
	from_int64_t_to_m256i_rns(avx_pbb, &rns_b, pbb);

	for (int i = 0; i < NTEST; i++)
	{

		avx_mult_mod_rns_cr_cox(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);
	}
	printf("Done.\n");

	// Testing
	printf("\tTesting... ");

	for (int i = 0; i < NSAMPLES; i++)
	{

		mpz_urandomm(A, state, modul_p); //Randomly generates A < P
		mpz_urandomm(B, state, modul_p); //Randomly generates A < P

		from_int_to_rns(pa, &rns_a, A);
		from_int_to_rns(pb, &rns_a, B);
		from_int_to_rns(pab, &rns_b, A);
		from_int_to_rns(pbb, &rns_b, B);

		from_int64_t_to_m256i_rns(avx_pa, &rns_a, pa);
		from_int64_t_to_m256i_rns(avx_pb, &rns_a, pb);
		from_int64_t_to_m256i_rns(avx_pab, &rns_b, pab);
		from_int64_t_to_m256i_rns(avx_pbb, &rns_b, pbb);

		for (int j = 0; j < NTEST; j++)
		{

			// RDTSC
			t1 = cpucyclesStart();

			avx_mult_mod_rns_cr_cox(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);

			t2 = cpucyclesStop();

			if (timing > (t2 - t1) / NFUNS)
				timing = (t2 - t1) / NFUNS;

			// Instructions
			before_instructions = rdpmc_instructions();

			avx_mult_mod_rns_cr_cox(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);

			after_instructions = rdpmc_instructions();

			if (instructions > (after_instructions - before_instructions) / NFUNS)
				instructions = (after_instructions - before_instructions) / NFUNS;

			// actual cycles
			before_cycles = rdpmc_actual_cycles();

			avx_mult_mod_rns_cr_cox(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);

			after_cycles = rdpmc_actual_cycles();

			if (cycles > (after_cycles - before_cycles) / NFUNS)
				cycles = (after_cycles - before_cycles) / NFUNS;

			// reference cycles
			before_ref = rdpmc_reference_cycles();

			avx_mult_mod_rns_cr_cox(avx_res, avx_pa, avx_pab, avx_pb, avx_pbb, &mult, tmp0, tmp1, tmp2, a);

			after_ref = rdpmc_reference_cycles();

			if (ref > (after_ref - before_ref) / NFUNS)
				ref = (after_ref - before_ref) / NFUNS;
		}
		fprintf(fpt, "\t\t\t{\n");
		fprintf(fpt, "\t\t\t\t\"TSC Cycles\" : %lld,\n\t\t\t\t\"Instructions\" : %ld,\n\t\t\t\t\"Actual cycles\" : %ld,\n\t\t\t\t\"Reference cycles\" : %ld\n\t\t\t}", timing, instructions, cycles, ref);
		if (i < NSAMPLES - 1)
			fprintf(fpt, ",");
		fprintf(fpt, "\n");
	}

	fprintf(fpt, "\t\t]\n\t}\n}");

	printf("Done.\n");
	printf("\tRNS parallel cox modular multiplication : %lld CPU cycles.\n", timing);
	printf("\tRNS parallel cox modular multiplication : %ld instructions.\n", instructions);
	printf("\tRNS parallel cox modular multiplication : %ld actual CPU cycles.\n", cycles);
	printf("\tRNS parallel cox modular multiplication : %ld reference CPU cycles.\n", ref);

	timing = ULLONG_MAX;
	cycles = ULONG_MAX;
	instructions = ULONG_MAX;
	ref = ULONG_MAX;

	fclose(fpt);

	return 0;
}
