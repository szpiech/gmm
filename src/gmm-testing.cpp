#include <iostream>
#include <fstream>
#include <cmath>
#include "gmm.h"

using namespace std;

int main (int argc, char* argv[])
{
  
  ifstream fin;
  fin.open(argv[1]);
  if(fin.fail())
    {
      cerr << "Failed to open test.points.mix\n";
      return -1;
    }
  

  int startPos = fin.tellg();

  int numPoints = 0;
  while(!fin.eof())
    {
      double junk;
      fin >> junk;
      if(!fin.eof()) numPoints++;
    }
  
  cerr << numPoints << "\n";

  double *data = new double[numPoints];

  fin.clear(); //remove eof flag

  fin.seekg(startPos); //back to the start of the file

  
  for(int i = 0; i < numPoints; i++)
    {
      fin >> data[i];
    }
 
  const int gaussians = 2;
  const size_t maxIterations = 250;
  const double tolerance = 1e-10;
  // const bool forcePositive = true;

  
  double *W = new double[2];
  W[0] = 0.8;
  W[1] = 0.2;

  double *Mu = new double[2];
  Mu[0] = 0.5;
  Mu[1] = 2;

  double *Sigma = new double[2];
  Sigma[0] = 0.03;
  Sigma[1] = 4;

  GMM gmm(gaussians,W,Mu,Sigma,maxIterations,tolerance,true,true);

  gmm.estimate(data,numPoints);


  /*
  gmm.x = data;
  gmm.M = numPoints;
  gmm.W = W;
  gmm.Mu = Mu;
  gmm.Sigma = Sigma;
  */
  /*
  for(int k=0; k < maxIterations; k++)  //EM iterations
	{
          double lhood = gmm.lhood();

	  cerr << lhood << endl;
	  gmm.W_update();
	  gmm.Mu_update();
	  gmm.Sigma_update();
	  gmm.update();

	  cout << "W = " << gmm.W[0] << " " << gmm.W[1] << endl;
	  cout << "Mu = " << gmm.Mu[0] << " " << gmm.Mu[1] <<endl;
	  cout << "Sigma = " << gmm.Sigma[0] << " " << gmm.Sigma[1] <<endl;
	  cout << "Maximum Likelihood=" <<gmm.lhood()<<endl;
	  cout << "BIC="<<gmm.BIC() <<endl;
	  cout <<"interations="<<k<< endl; 


          if(log(gmm.lhood())-log(lhood)<= tolerance) 
	    {
	      //cout << "------ Fit with one Gaussian Mixtures----" << endl;
	      
	      break;
	    }
	}     
  */


  /*
  for(int i = 1; i < 2; i++)
    {
      cerr << i << " ";
      EMFit<> em(maxIterations, tolerance, forcePositive);
      
      // Calculate mixture of Gaussians.
      GMM<> gmm(size_t(i), dataPoints.n_rows, em);
      
      // Compute the parameters of the model using the EM algorithm.
      Timer::Start("em");
      double likelihood = gmm.Estimate(dataPoints, 10);
      Timer::Stop("em");
      
      cerr << " " << likelihood << endl;
    }
  */


  //gmm.Save("test.output.gmm");


  return 0;
}
