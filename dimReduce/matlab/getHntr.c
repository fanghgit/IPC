#include <math.h>
#include "mex.h"

void mexFunction(
        int nlhs,       mxArray *plhs[],
        int nrhs, const mxArray *prhs[]
        )
{
	/* evaluate norm(X - BH, 'fro')^2 by for loop  */
	int m, n, r;
	if(nrhs != 5) mexErrMsgTxt ("must have 5 inputs");
	
	double *Y = mxGetPr(prhs[0]);
	double *w = mxGetPr(prhs[1]);
	double ntr = *mxGetPr(prhs[2]);
	int k = (int)*mxGetPr(prhs[3]);
	double lambda1 = *mxGetPr(prhs[4]);
	
	printf("ntr = %f", ntr);
	printf("k = %i", k);
	
	/*
	n = mxGetN(prhs[0]);
	r = mxGetN(prhs[1]);
	double *X = mxGetPr(prhs[0]);
	double *B = mxGetPr(prhs[1]);
	double *H = mxGetPr(prhs[2]);
	
	double result = 0;
	double val = 0;
	for(int i = 0; i < m; i++){
		for(int j = 0; j < n; j++){
			val = X[j*m + i];
			for(int k = 0; k < r; k++){
				val = val - B[k*m + i]*H[j*r + k];
			}
			result += val*val;
		}
	}
	plhs[0] = mxCreateDoubleScalar( result ); */

}