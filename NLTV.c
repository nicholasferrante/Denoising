/*
 * NLTV.c - nonlocal denoising function
 *
 * u = NLTV(f,lambda);
 *
 * f is 2-d function.
 * lambda is positive scalar.
 * u is 2-d function.
 *
 * This is a MEX-file for MATLAB.
 *
 * Created:         10/04/10
 * Last Modified:   10/14/10
 * Author:          David Mao
 */


#include "mex.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define EPS 1e-3
#define INF 1e300

int     m, n;   /* size of the image */
int     rw, sw; /* size of the nonlocal window */

void	mainloop(const double *const f, const double lambda, double *const u);
void	TV_w(const double *const u, const int *const w, double *const T);
void	K_w(const double *const u, const int *const w, const double *const T, double *const K);
double	linesearch(double *const u, const double *const f, const int *const w, const double mu, double *const T, double *const K);
void	preweight(const double *const f, int *const w);
double	energy(const double *const f, const double *const u, const double *const T, const double mu);


/*  main subroutine */

void mainloop(const double *const f, const double lambda, double *const u){
    
    int t; 
    double mu, decrease, E, oldE;
	
	int     *w;
    double  *T;
    double  *K;
    
    rw = 4;										/* default radius of window */
    sw = rw * 2 + 1;							/* size of window */
    mu = lambda / (1 + lambda);					/* standardized parameter */
    
    w = calloc(m * n * sw * sw, sizeof(int));	/* weight */
    T = calloc(m * n, sizeof(double));			/* nonlocal total variation */
    K = calloc(m * n, sizeof(double));			/* nonlocal curvature */
    
    memcpy(u, f, m * n * sizeof(double));		/* u = f; */
    preweight(f, w);							/* calculate weight	*/
    TV_w(u, w, T);								/* calculate nonlocal TV */
	E = energy(f, u, T, mu);					/* calculate objective energy */

    decrease = INF;
	t = 0;
    while (decrease>EPS && t<50){
        oldE = E;
        t++;
        E = linesearch(u, f, w, mu, T, K);		
        decrease = 1 - E / oldE;
    }
    
    free(w);
    free(T);
    free(K);
    
}

/*  nonlocal TV */

void	TV_w(const double *const u, const int *const w, double *const T){
	
    int i, j, k1, k2, k1p, k2p;
    double uij, s, wdu;
    
	/* moving pointer */
    const double    *up = u; 
    const int       *wp = w; 
    double          *Tp = T; 
    
    for (j = 0; j < n; j++){
        for (i = 0; i < m; i++){
            uij = *up;
            s = 0;
            for (k2 = -rw; k2 <= rw; k2++){
				k2p = (j + k2) >= 0 ? ((j + k2) < n ? k2 : (n - 1 - j)) : -j;		/* make sure 0 <= j + k2p < n */
                for (k1 = -rw; k1 <= rw; k1++){
					k1p = (i + k1) >= 0 ? ((i + k1) < m ? k1 : (m - 1 - i)) : -i;	/* make sure 0 <= i + k1p < m */
                    wdu = (*(up + m * k2p + k1p) - uij) * *wp;
                    s += wdu * wdu;
					wp++;
                }
            }
			s += EPS;
            *Tp = sqrt(s); 
			up++;
			Tp++;
        }
    }
    up = u; wp = w; Tp = T; 
}

/*  nonlocal curvature */

void K_w(const double *const u, const int *const w, const double *const T, double *const K){
    
    int i, j, k1, k2, k1p, k2p;
    double uij, Tij, s, wdu;
    
	/* moving pointer */
    const double    *up = u; 
    const int       *wp = w; 
    const double    *Tp = T; 
    double          *Kp = K; 
    
    for (j = 0; j < n; j++){
        for (i = 0; i < m; i++){
            uij = *up;
            Tij = *Tp;
            s = 0;
            for (k2 = -rw; k2 <= rw; k2++){
				k2p = (j + k2) >= 0 ? ((j + k2) < n ? k2 : (n - 1 - j)) : -j;		/* make sure 0 <= j + k2p < n */
                for (k1 = -rw; k1 <= rw; k1++){
					k1p = (i + k1) >= 0 ? ((i + k1) < m ? k1 : (m - 1 - i)) : -i;	/* make sure 0 <= i + k1p < m */
                    wdu = (*(up + m * k2p + k1p) - uij) * *wp; 
                    s += wdu * (1 / (Tij + EPS) + 1 / (*(Tp + m * k2p + k1p) + EPS));
					wp++;
                }
            }
			*Kp = s; 
            up++; Tp++; Kp++; 
        }
    } 
	up = u; wp = w; Tp = T; Kp = K; /* reset moving pointer */
}

/*  gradient descent line search */

double linesearch(double *const u, const double *const f, const int *const w, const double mu, double *const T, double *const K){
    
    int i, j;
    double dt, E, oldE, decrease;
    
    double          *const tu=(double *)calloc(m*n, sizeof(double));	/* temporary u */
    double          *const du=(double *)calloc(m*n, sizeof(double));	/* forward step */
    
	/* moving pointer */
    double          *up = u; 
    const double    *fp = f; 
    const double    *Kp = K; 
    double          *tup = tu; 
    double          *dup = du; 
    
    dt = 1;
    TV_w(u, w, T);
    K_w(u, w, T, K);
    
	/* du = K * (1 - mu) + (f - u) * mu; */
    for (j = 0; j < n; j++){
        for (i = 0; i < m; i++){
            *dup = *Kp * (1 - mu) + (*fp - *up) * mu;
            dup++; Kp++; fp++; up++;
        }
    }
    dup = du; Kp = K; fp = f; up = u; 
    
	/* tu = u + dt * du; */
    for (j = 0; j < n; j++){
        for (i = 0; i < m; i++){
            *tup = *up + dt * *dup;
            tup++; up++; dup++;
        }
    }
    tup = tu; up = u; dup = du; 
    

    TV_w(tu, w, T);
	
	E = energy(f, tu, T, mu);

    decrease = 0;
    while ((decrease>=0)  &&  (dt>0.01)){
		
        oldE = E;
        dt = dt / 2;
		
		/* tu = u + dt * du; */
        for (j = 0; j < n; j++){
			for (i = 0; i < m; i++){
                *tup = *up + dt * *dup;
                tup++; up++; dup++;
            }
        }
        tup = tu; up = u; dup = du; 
		
        TV_w(tu, w, T);
		E = energy(f, tu, T, mu);
        decrease = oldE - E;
    }
    
	/* u = u + dt * 2 * du; the second last one is the best */
    for (j = 0; j < n; j++){
        for (i = 0; i < m; i++){
            *up += dt * 2 * *dup;
            up++; dup++;
        }
    }
    up = u; dup = du;  
    
    free(du);
    free(tu);
    
    E = oldE;
    return E;
}

/*  evaluate energy */

double energy(const double *const f, const double *const u, const double *const T, const double mu){
	
	int i,j;
	double E, E1, E2;
	
	/* moving pointer */
	const double	*up = u; 
    const double    *fp = f; 
    const double    *Tp = T; 
	
	E1 = 0;
	E2 = 0;
	
    for (j = 0; j < n; j++){
        for (i = 0; i < m; i++){
            E1 += *Tp;							/* nonlocal total variation */
			E2 += (*up - *fp) * (*up - *fp);	/* fidelity */
            up++; fp++; Tp++;
        }
    }
	up = u; fp = f; Tp = T;						/* reset moving pointer */
	
	E = E1 * (1 - mu) + E2 * mu / 2;
	return E;
}

/*  binary weight */

void preweight(const double *const f, int *const w){
	
    int i, j, k1, k2, k1p, k2p, l1, l2, mink1, mink2, s;
    double fij, g, minbar;
	
    const int rp = 2;		/* default radius of patch */
    const int sim = 4;		/* number of most similar patches */
    const int num_nbr = 5;	/* number of neighbours */
    int nbr[] = {
		sw * (rw - 1) + rw, 
        sw * rw + rw - 1,
        sw * rw + rw,
        sw * rw + rw + 1,
        sw * (rw + 1) + rw}; /* 4-neighbours */
    
    double  *const	d = (double *)calloc(m*n*sw*sw, sizeof(double));	/* difference between patches */
    double  *const	window = (double *)calloc(sw*sw, sizeof(double));	/* moving window */
	int  *const		ind_1 = (int *)calloc(sw*sw, sizeof(int));			/* coordinates of the window */
	int  *const		ind_2 = (int *)calloc(sw*sw, sizeof(int));			/* coordinates of the window */
    
	/* moving pointer */
    const double    *fp = f; 
    int             *wp = w;
    double          *dp = d;
	double			*windowp = window;
	int				*ind_1p = ind_1;
	int				*ind_2p = ind_2;
	double			*minp;
    
    for (j = 0; j < n; j++){
        for (i = 0; i < m; i++){
            fij = *fp;
            for (k2 = -rw; k2 <= rw; k2++){
				k2p = (j + k2) >= 0 ? ((j + k2) < n ? k2 : (n - 1 - j)) : -j;								/* make sure 0 <= j + k2p < n */
                for (k1 = -rw; k1 <= rw; k1++){
					k1p = (i + k1) >= 0 ? ((i + k1) < m ? k1 : (m - 1 - i)) : -i;							/* make sure 0 <= i + k1p < m */
                    g = (*(fp + m * k2p + k1p) - fij) * (*(fp + m * k2p + k1p) - fij);						/* difference between pixels */
                    for (l2 = -((j - rp) >= 0 ? rp : j); l2 <= ((j + rp) < n ? rp : (n - 1 - j)); l2++){	/* inside the boundary */
                        for (l1 = -((i - rp) >= 0 ? rp : i); l1<=((i + rp) < m ? rp : (m - 1 - i)); l1++){	/* inside the boundary */
                            *(dp + (m * l2 + l1) * sw * sw) += g;											/* spread the difference to neighbours like convolution */
                        }
                    }
                    *dp += EPS;
                    dp++;
                }
            }
            fp++;
        }
    }
    fp = f; dp = d; 
	
	/* calculate the coordinates of the small window */
	for (k2 = -rw; k2 <= rw; k2++){
		for (k1 = -rw; k1 <= rw; k1++){
			*ind_1p = k1; ind_1p++;
			*ind_2p = k2; ind_2p++;
		}
	}
	ind_1p = ind_1; ind_2p = ind_2; 
	
    for (j = 0; j < n; j++){
        for (i = 0; i < m; i++){
            
			memcpy(window, dp, sw * sw * sizeof(double));	/* cut a window */
			
			for (s = 0; s < num_nbr; s++){
				*(window + nbr[s]) = INF;					/* neighbours don't count in comparison */
				*(wp + nbr[s]) = 1;							/* weight of neighbours are set */
			}
			
            for (s = 0; s < sim; s++){
				minbar = INF;
				for (k2 = -rw; k2 <= rw; k2++){
					for (k1 = -rw; k1 <= rw; k1++){
						if (*windowp < minbar){
							minbar = *windowp;
							minp = windowp;
						}
						windowp++;
                    }
                }
				windowp = window;							/* position of the s-th smallest one */
				*minp = INF;
				*(wp + (minp - window)) = 1;				/* weight of the s-th smallest one */
				
				/* make w being symmetric */
				mink1 = *(ind_1 + (minp - window));
				mink2 = *(ind_2 + (minp - window));
				if (((j + mink2) >= 0) && ((j + mink2) < n) && (i + mink1) >= 0 && (i + mink1 < m)){	/* inside the boundary */
					*(wp + (m * mink2 + mink1) * sw * sw + sw * (rw - mink2) + rw - mink1) = 1;			/* the mirror guy is set as well */
				}
            }
            
            dp += sw*sw;
            wp += sw*sw;
        }
    }
    dp = d; wp = w; 
	
    free(d);
    free(window);
    free(ind_1);
	free(ind_2);
}

/* the gateway function */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    double *f = mxGetPr(prhs[0]);
    double lambda = mxGetScalar(prhs[1]);
    
    m = mxGetM(prhs[0]);
    n = mxGetN(prhs[0]);
    
    plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
    double *u = mxGetPr(plhs[0]);
    
    /*  call the C subroutine */
    mainloop(f, lambda, u);
    
}

