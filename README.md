# RMTEstimCovDist
Code that gives the RMT Improved distances between covariances.

The code contains the following scripts and functions

A function RMTCovDistEst that takes as input the samples X and Y (of the two distribution from which the distance will be computed) and the type of distance (Fisher,Battacharrya,KL,Wasserstein) and returns the proposed estimates est as well as the classical estimate esthat.

A function RMT_estim that takes as input the known covariance matrix S, the sample covariance Cs of the unknown covariance matrix C the sample size n2 and the distance. It returns the estimate of the distance where one of the two matrices are known.

A script compareEst which compares the proposed estimates and the classical one.

A script Spectral_clustering_Fisher which performs the clustering over covariances.

Feel free to contact tiomoko_malik@yahoo.fr for further details
