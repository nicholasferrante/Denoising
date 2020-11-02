/*
 * PushForward.c - pushforward function
 * 
 * v=PushForward(u,vecX,vecY);
 *
 * u is 2-d function.
 * [vecX,vecY] is 2-d vector field.
 * v is 2-d function.
 *
 * This is a MEX-file for MATLAB.
 * Created:         02/09/10
 * Last Modified:   08/26/10
 * Author:          David Mao
 */

#include "mex.h"
#include <math.h>


/*  main subroutine */
void push(double *u, double *vecX, double *vecY, double *v, mwSize m, mwSize n) {
    mwIndex i, j = 0;
    for (j=0; j<n; j++) {
        for (i=0; i<m; i++) {
            
            /*  linear index */
            mwIndex ind = i+j*m;
            
            /*  u(i,j) */
            double uij = *(u+ind);
            
            /*  the destination of pixel (i,j) after push */
            double x = j+ *(vecX+ind);
            double y = i+ *(vecY+ind);
            
            mwIndex intx = floor(x);
            mwIndex inty = floor(y);
            
            double wx = x-intx;
            double wy = y-inty;
            
            /*  calculate v(x,y) */
            *(v+inty+intx*m) += (1-wx)*(1-wy)*uij;
            *(v+(inty+1)+intx*m) += (1-wx)*(wy)*uij;
            *(v+inty+(intx+1)*m) += (wx)*(1-wy)*uij;
            *(v+(inty+1)+(intx+1)*m) += (wx)*(wy)*uij;
            
        }
    }
}

/* the gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    double *u = mxGetPr(prhs[0]);
    double *vecX = mxGetPr(prhs[1]);
    double *vecY = mxGetPr(prhs[2]);
    
    mwSize m = mxGetM(prhs[0]);
    mwSize n = mxGetN(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    
    double *v = mxGetPr(plhs[0]);
    
    /*  call the C subroutine */
    push(u, vecX, vecY, v, m, n);
    
}

