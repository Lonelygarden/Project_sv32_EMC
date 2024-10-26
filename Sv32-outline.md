# Exact MC:

1. Draw the terminal variance $V_T$ given $V_0$ by non-central chi-squared RV.
2. Using the numerical Laplace inversion to get the CDF of Integrated Variance.
   * Other method: Gamma Expansion
3. Find the root of U=CDF(X) numerically.

# Almost Exact MC:

1. Draw the terminal variance  $V_T$ given $V_0$ by non-central chi-squared RV.

2. Using other method to get the CDF of Integrated Variance, avoiding calculating Laplace inversion

   * IG Approximation

     Draw the parameters for IG approximation, here are several ways:

     * draw the mean, variance from Laplace transform
     * draw the mean, variance and skew from Laplace transform
     * draw the mean, variance from analytic formula in Tse and Wan (2013)

   * Poisson Gamma Expansion

3. Draw the Integrated Variance from the distribution and simulate.