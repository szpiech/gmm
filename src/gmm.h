/*
  NAME:
    gmm.h

  PURPOSE:
    A class that implements a 1-dimensional Gaussian Mixture Model fit with the EM algorithm

  PLATFORM:
    Tested in 2013 on a MacPro running OSX 10.8.2, but it should be platform independent as long as GSL is available.
  
  DEPENDENCIES:
    Requires GNU GSL, which can be found at <http://www.gnu.org/software/gsl/>.
    When compiling, use the flags 
      -lgsl -lgslcblas 
    or 
      $(LIB_PATH)/libgsl.a $(LIB_PATH)/libgslcblas.a

  USAGE:
    The class object contains all the machinery to do a GMM estimation with the EM algorithm.
    Upon instantiation, GMM will require the following:
    
      n : number of Gaussians to use

      a : array of initial guesses for the mixture coefficients

      mean :  array of intial guesses for the means

      var :  array of initial guesses for the variances

    Optional parameters:

      maxIter : maximum number of iterations of the EM algorithm, default 250
      
      p : desired precision stopping condition, default 1e-5
      
      v : if true, will output progress of each step of EM algorithm, default true

    To run the EM algorithm, call GMM::estimate(double *data, int dataSize)

    Example:
    
      GMM gmm(n,a,mean,var);
      gmm.estimate(data,dataSize);


  Copyright (C) 2013  Zachary A Szpiech (szpiech@gmail.com)

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
  
 */

#include <iostream>
#include <iomanip>
#include <gsl/gsl_randist.h>
#include <cmath>
#include <limits>
#include <ctime>

using namespace std;

class GMM
{
 private:
  
  int numGaussians; //How many gaussians do we assume?
  double *a; //Mixture proportions
  double *mean;
  double *var;
  //These hold the values for the next update
  double *a_t; 
  double *mean_t;
  double *var_t;

  int dataSize;
  double *x; //a pointer to the data of length dataSize

  int maxIterations; //EM will stop after this many iterations if it hasn't converged
  double precision; //Convergence condition

  bool verbose; //if true, prints information to stderr

  double loglikelihood; //of the data given the given model and current parameters
  double BIC; //Bayseian Information Criteria for the data given the given model and current parameters

  void update(); //Update parameters, this folds both the E and M step into a single calculation
  double normal(double x, double mean, double var);
  void lhood(); //calculate the log likelihood and BIC
  void printState();
  
 public:

  GMM(int n, double* a_init, double* mean_init, double* var_init, int maxIt, double p, bool v);
  ~GMM();

  bool estimate(double* data, int size);

  double getBIC();
  double getLogLikelihood();
  double getMixCoefficient(int i);
  double getMean(int i);
  double getVar(int i);

};


inline double GMM::getMixCoefficient(int i)
{
  if(i >=0 && i <= numGaussians && a != NULL)
    {
      return a[i];
    }
  else
    {
      cerr << "ERROR: out of bounds.\n";
      throw 1;
    }
}

inline double GMM::getMean(int i)
{
  if(i >=0 && i <= numGaussians && mean != NULL)
    {
      return mean[i];
    }
  else
    {
      cerr << "ERROR: out of bounds.\n";
      throw 1;
    }
}

inline double GMM::getVar(int i)
{
  if(i >=0 && i <= numGaussians && var != NULL)
    {
      return var[i];
    }
  else
    {
      cerr << "ERROR: out of bounds.\n";
      throw 1;
    }
}

inline double GMM::getBIC()
{
  return BIC;
}

inline double GMM::getLogLikelihood()
{
  return loglikelihood;
}

/*
  Calculates the log likelihood and BIC of the data given the model and current parameters
  
  INPUT:
  
    none, but assumes the model parameters have been initialized and the data pointer is vaild

  OUTPUT:

    none

  FUNCTION:

    Parameters BIC, and loglikelihood are modified.

    Log Likelihood function:
    
      l = \sum_{j=1}^N ln( \sum_{k=1}^K a_k p_k(x_j|\theta_k) )

    BIC:

      -2l + (3K-1)ln(N)

    Where:

      k : indexes Gaussian components
      K : total number of Gaussians
      j : indexes data
      N : total number of data points
      x_j : jth element of the data array x
      a_k : mixture coefficient for Gaussian k
      \theta_k : parameter vector (\mu_k,\Sigma_k) for Gaussian k
      \mu_k : mean of Gaussian k
      \Sigma_k : variance of Gaussian k

 */
inline void GMM::lhood()
{
  double sum1, sum2;
  
  sum2 = 0;
  for (int j = 0; j < dataSize; j++)
    {
      sum1 = 0;
      for (int k = 0; k < numGaussians; k++)
	{ 
	  sum1 += a[k]*normal(x[j],mean[k],var[k]);
	}
      sum2 += log(sum1);
    }

  loglikelihood = sum2;
  BIC = -2.0*loglikelihood+double(3.0*numGaussians-1)*log(dataSize);
  
  return;
}

inline GMM::GMM(int n, double* a_init, double* mean_init, double* var_init, int maxIt = 250, double p = 1e-5, bool v = true)
{
  if(mean_init == NULL || var_init == NULL || a_init == NULL)
    {
      cerr << "ERROR: NULL pointer passed as initial value.\n";
      throw 1;
    }

  if(n < 1)
    {
      cerr << "ERROR: Can not have fewer than 1 Gaussian.\n";
      throw 1;
    }

  numGaussians = n;

  a = new double [numGaussians];
  mean = new double [numGaussians];
  var = new double [numGaussians];

  for (int i = 0; i < numGaussians; i++)
    {
      a[i] = a_init[i];
      mean[i] = mean_init[i];
      var[i] = var_init[i];
    }  

  a_t = new double [numGaussians];
  mean_t = new double [numGaussians];
  var_t = new double [numGaussians];

  x = NULL;
  maxIterations = maxIt;
  precision = p;
  verbose = v;

  loglikelihood = numeric_limits<double>::min();
  BIC = numeric_limits<double>::max();

  return;
}

inline GMM::~GMM()
{
  x = NULL;
  delete [] a;
  delete [] a_t;
  delete [] mean;
  delete [] mean_t;
  delete [] var;
  delete [] var_t;
  return;
}

inline double GMM::normal(double x, double mean, double var)
{
  return gsl_ran_gaussian_pdf(x-mean,sqrt(var));
}


/*
  Calculates one step of the EM algorithm
  
  INPUT:
  
    none, but assumes the model parameters have been initialized and the data pointer is vaild

  OUTPUT:

    none

  FUNCTION:

    Performs both the expectation and maximization steps simultaneously, then calculates the
    log likelihood and BIC for the current model and parameters.  The calculation has been
    broken down in such a way as to minimize the number of loops needed (or at least get close
    to the minimum).
    
    Parameters a, mean, var, a_t, mean_t, var_t, BIC, and loglikelohood are modified.

    E-step:
    
      w^t_{jk} = a^t_k p_k(x_j|\theta^t_k) / ( \sum_{i=1}^K a^t_i p_i(x_j|\theta^t_i) )

    M-step:

      a^{t+1}_k = 1/N \sum_{j=1}^N w^t_{jk}

      \mu^{t+1}_k = ( \sum_{j=1}^N w^t_{jk} x_j ) / ( \sum_{j=1}^N w^t_{jk} )

      \Sigma^{t+1}_k = ( \sum_{j=1}^N w^t_{jk} (x_j - \mu^{t+1}_k)^2 ) / ( \sum_{j=1}^N w^t_{jk} )

    Where:

      t : current iteration
      k : indexes Gaussian components
      K : total number of Gaussians
      j : indexes data
      N : total number of data points
      x_j : jth element of the data array x
      w_{jk} : prob of membership of data point j in gaussian k
      a_k : mixture coefficient for Gaussian k
      p_k(x_j|\theta_k) : PDF of kth Gaussian with parameters \theta_k
      \theta_k : parameter vector (\mu_k,\Sigma_k)
      \mu_k : mean of Gaussian k
      \Sigma_k : variance of Gaussian k
      

 */
inline void GMM::update()
{
  double wjk_num, wjk_den, sum_wjk, sum_wjk_xj, sum_wjk_xj2;
  for (int k = 0; k < numGaussians; k++)
    {
      sum_wjk = 0;
      sum_wjk_xj = 0;
      sum_wjk_xj2 = 0;

      for (int j = 0; j < dataSize; j++)
	{
	  wjk_num = a[k]*normal(x[j],mean[k],var[k]);
	  
	  wjk_den = 0;
	  for (int i = 0; i < numGaussians; i++) wjk_den += a[i]*normal(x[j],mean[i],var[i]);

	  sum_wjk += wjk_num/wjk_den;
	  sum_wjk_xj += x[j]*wjk_num/wjk_den;
	  sum_wjk_xj2 += x[j]*x[j]*wjk_num/wjk_den;
	  
	}

      a_t[k] = sum_wjk/double(dataSize);
      mean_t[k] = sum_wjk_xj/sum_wjk;
      var_t[k] = sum_wjk_xj2/sum_wjk - mean_t[k]*mean_t[k];
    }

  //Assign next iteration parameters to current parameters
  for (int k = 0; k < numGaussians; k++)
    {
      a[k] = a_t[k];
      mean[k] = mean_t[k];
      var[k] = var_t[k];
    }

  lhood();//updates likelihood and BIC

  return;
}


/*
  Prints the current values of all parameters, log likelihood, and BIC
 */
inline void GMM::printState()
{
  cerr << setprecision(5) << scientific;

  cerr << "(";
  for(int k=0;k<numGaussians-1;k++) cerr << a[k] << ",";
  cerr << a[numGaussians-1] << ")\t";
  
  cerr << "(";
  for(int k=0;k<numGaussians-1;k++) cerr << mean[k] << ",";
  cerr << mean[numGaussians-1] << ")\t";
  
  cerr << "(";
  for(int k=0;k<numGaussians-1;k++) cerr << var[k] << ",";
  cerr << var[numGaussians-1] << ")\t";

  cerr << loglikelihood << "\t" << BIC << endl;
}


/*
  Starts the GMM EM estimation proceedure
  
  INPUT:
  
    double *data : an array of data point observations to use for GMM
    int size : the length of the array

  OUTPUT:

    bool : true if the EM algorithm converged to the specified precision
           false otherwise

  FUNCTION:

    Executes EM steps until convergence or until reached maxIterations.
    Parameters a, mean, var, a_t, mean_t, var_t, BIC, and loglikelohood are modified.

 */
inline bool GMM::estimate(double *data, int size)
{

  bool converged = false;

  if(data == NULL || size < 1)
    {
      cerr << "Invalid dataset.\n";
      throw 1;
    }

  if(verbose) cerr << "Begin GMM estimation with k = " << numGaussians << " Gaussians...\n";

  dataSize = size;
  x = data;

  lhood();

  double lastloglikelihood = loglikelihood;

  if(verbose)
    {
      cerr << "iteration\tmixture\tmean\tvar\tlogL\tBIC\n";
      cerr << "0\t";
      printState();
    }

  for(int i = 1; i <= maxIterations; i++)
    {
      //EM steps are done here, includes recalculation of the likelihood and BIC
      update(); 

      if(verbose)
	{
	  cerr << i << "\t";
	  printState();
	}

      if(abs(loglikelihood - lastloglikelihood) <= precision)
	{
	  converged = true;
	  break;
	}

      lastloglikelihood = loglikelihood;
    }

  return converged;
}


/*
 *  IMPORTANT! This method does NOT clean up after itself
 *  on the assumption that these data will be used after the
 *  destruction of the object.
 */
class GMMParameters
{
 public:

  GMMParameters(double* data, int size, int num);
  ~GMMParameters();
  double* getMixtures();
  double* getMeans();
  double* getVariances();
  void newMeanGuess();

 private:
  const gsl_rng_type * T;
  const gsl_rng * r;

  int n;
  double* x;
  
  int k;
  double *a;
  double *mean;
  double *var;
};

inline GMMParameters::GMMParameters(double* data, int size, int num)
{
  if(data == NULL || num < 1)
    {
      cerr << "Invalid dataset.\n";
      throw 1;
    }

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, time(NULL));

  n = size;
  x = data;
  k = num;

  a = new double [k];
  mean = new double [k];
  var = new double [k];

  double v = gsl_stats_variance(x,1,n);

  for(int i = 0; i < k; i++)
    {
      a[i] = 1.0/double(k);
      mean[i] = 0;
      var[i] = v;
    }
  
  newMeanGuess();
  
  return;
}

inline GMMParameters::~GMMParameters()
{
  x = NULL;
  a = NULL;
  mean = NULL;
  var = NULL;
  gsl_rng_free(r);

  return;
}

inline void GMMParameters::newMeanGuess()
{
  gsl_ran_sample(r,mean,k,x,n,sizeof(double));
  return;
}

inline double* GMMParameters::getMixtures()
{
  return a;
}

inline double* GMMParameters::getMeans()
{
  return mean;
}

inline double* GMMParameters::getVariances()
{
  return var;
}
