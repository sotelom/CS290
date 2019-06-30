#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <ctime>
#include <sys/time.h>
#include <sys/resource.h>

#define SSE_WIDTH	4

#define ALIGNED		__attribute__((aligned(16)))

#ifndef ASIZE
    #define ASIZE        8*1024*1024
#endif

#ifndef NUMTRIES
    #define NUMTRIES        100
#endif



void SimdMul(float *a, float *b, float *c, int len)
{
	int limit = ( len/SSE_WIDTH ) * SSE_WIDTH;  // SSE_WIDTH is the # of floats processed by a single SSE instruction
	__asm
	(
		".att_syntax\n\t"
		"movq    -24(%rbp), %r8\n\t"		// a, move quadword (64b/8B) address &a[0] into r8 register
		"movq    -32(%rbp), %rcx\n\t"		// b, move quadword (64b/8B) address &b[0] into rcx register
		"movq    -40(%rbp), %rdx\n\t"		// c, move quadword (64b/8B) address &c[0] into rdx register
	);

	for( int i = 0; i < limit; i += SSE_WIDTH )
	{
		__asm
		(
			".att_syntax\n\t"
			"movups	(%r8), %xmm0\n\t"	// load the first sse register,  [r8]  = a[SSE_WIDTH*i to SSE_WIDTH*i+SSE_WIDTH-1] into xmm0
			"movups	(%rcx), %xmm1\n\t"	// load the second sse register, [rcx] = b[SSE_WIDTH*i to SSE_WIDTH*i+SSE_WIDTH-1] into xmm1
			"mulps	%xmm1, %xmm0\n\t"	// do the multiply, xmm0[i] = xmm0[i]*xmm1[i], for i = 0 to SSE_WIDTH*i to SSE_WIDTH-1
			"movups	%xmm0, (%rdx)\n\t"	// store the result, move xmm0 into address stored in rdx = c[SSE_WIDTH*i to SSE_WIDTH*i+SSE_WIDTH-1]
			"addq $16, %r8\n\t"         // Increment all addresses by sizeof(float)*SSE_WIDTH = 16
			"addq $16, %rcx\n\t"
			"addq $16, %rdx\n\t"
		);
	}

	for( int i = limit; i < len; i++ ) // Do the remainder of the values that are not a multiple of SSE_WIDTH
		c[i] = a[i] * b[i];
}



int main(int argc, char *argv[])
{
    float *x = new float [ASIZE];
    float *y = new float [ASIZE];
    float *z = new float [ASIZE];
    
    double avgGigaMults = 0.;
    double maxGigaMults = 0.;   
    
     for (int t = 0; t < NUMTRIES; t++)
    {
        double time0 = omp_get_wtime();
        
        SimdMul(x, y, z, ASIZE);
        
        double time1 = omp_get_wtime();

        double gigaMults = (double)(ASIZE) / (time1 - time0) / 1000000000.;
        avgGigaMults += gigaMults;
        if (gigaMults > maxGigaMults)
            maxGigaMults = gigaMults;
    }
    avgGigaMults /= (double)NUMTRIES;
    fprintf(stdout, "%12.6lf\t%12.6lf\n", maxGigaMults, avgGigaMults);
    
    avgGigaMults = 0.;
    maxGigaMults = 0.;   
    
     for (int t = 0; t < NUMTRIES; t++)
    {
        double time0 = omp_get_wtime();
        
        #pragma omp simd safelen(4) simdlen(4) aligned(x:16) aligned(y:16) aligned(z:16)
        for (int i=0; i<ASIZE; i++)
        {
            z[i] = x[i]*y[i];
        }                
        
        double time1 = omp_get_wtime();
        

        double gigaMults = (double)(ASIZE) / (time1 - time0) / 1000000000.;
        avgGigaMults += gigaMults;
        if (gigaMults > maxGigaMults)
            maxGigaMults = gigaMults;
    }
    avgGigaMults /= (double)NUMTRIES;
    fprintf(stdout, "%12.6lf\t%12.6lf\n", maxGigaMults, avgGigaMults);    
        
    
    return 0;       
}
