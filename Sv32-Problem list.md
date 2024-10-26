# ExactMC Problems and Tests：

1. In case IV(ATM and near expireation), the results are nan.

   **Location:** func iv_complex(nu, zz), here the parameter **nu** is extermely large, 

   max(nu)=(2.7673588380752325e+23-2.7673588380752325e+23j)

   min(nu) = (23742798775.766956-23742798775.766956j)

   

   thus in Line 54,  

   p0 = np.power(0.5 * zz, nu) / spsp.gamma(nu + 1)

   np.power(0.5 * zz, nu) and spsp.gamma(nu + 1) are all nan

   

   this may caused by the small texp and atm property

   **preliminary solutions:**

   1. using mpmath package to make dps higher
   2. try to check the method to power operations

2. In case V(near expiration only), the results are nan.

   (When texp = 0.001

   **Location:** func iv_complex(nu, zz), in Line 57

   for kk in np.arange(1, 64):

   ​      p0 *= zzh2 / (kk * (kk + nu))

   ​      iv += p0

   when kk = 44,  p0 becomes nan

   so when getting **m1, var** by taking numerical derivatives, we get nan.)

   

   When texp = 0.1, 

   **Location:** func iv_complex(nu, zz), here the parameter **nu** is extermely large, 

   max(nu)=(2.7673588380752325e+23-2.7673588380752325e+23j)

   min(nu) = (23742798775.766956-23742798775.766956j)

   thus in Line 54,  

   p0 = np.power(0.5 * zz, nu) / spsp.gamma(nu + 1)

   np.power(0.5 * zz, nu) and spsp.gamma(nu + 1) are all nan

   

   this is merely caused by the small texp

   **preliminary solutions:**

   same as case IV.

   **comments:**

   Since in the same model, if only strike prices are different, the parameters such as nu, are the same. So we get the same problem in Case VI and Case V.

3. In case VI(atm only), the results are nan.

   **Location:** func cond_spot_sigma(self, texp, var_0), in line 148

   sigma_cond = np.sqrt((1.0 - self.rho**2) * avgvar / var_0)

   Here avgvar has some negative values, hence taking square will generate nan.

   max(avgvar) = 6.29786861085193

   min(avgvar) = -2.558647429974919

   The process of finding root can generate negative avgvar.

   **preliminary solutions:**

   1. Exclude the negative points
   2. increase the precision of finding the root
   3. increase the precision of getting inverse CDF (try uniform distribution)

   **comments:**

   In the ATM situation, the problem only appears in the ending part--draw the values of average variance form the distribution.

# Almost Exact MC Problems and Tests:

Since Almost Exact MC simulation inherits the most of methods of Exact MC, some problems may be the same.

1. In Case I, the bias can be large, depending on dx when taking derivative.

   **Location:**  function cond_avgvar_mv_numeric(self, dt, var_0, var_t), in line 413,414, if we set dx<=1e-6, the bias is significantly large,

   if dx=1e-5, bias = -0.006352854569449451

   if dx=1e-6, bias = 0.55836720085399

   if dx=1e-7, bias = 446.30619326714344

   if dx=1e-8, bias = 544.4652564744149

   if dx=1e-9, bias = 656.4209183168899

   Here we compare the mean and var drew from taking derivative of cumulant-generating function:

   when dx=1e-5,  min(mean)=1.1362073668549382, max(mean)=1.5578891141593711,

   min(var)=0.0026287905661350327, max(var)=0.005852542258263123

   when dx=1e-6,  min(mean)=1.136207320932359, max(mean)=1.5578890681100175,

   min(var)=0.03690701179847651, max(var)=0.05130475374557833

   when dx=1e-7,  min(mean)=1.1362068820982205, max(mean)=1.5578885984949573,

   min(var)=-0.45283173931590626, max(var)=0.38700786699149303

   when dx=1e-8,  min(mean)=1.136208433819926, max(mean)=1.557890347281241, 

   min(var)=165.3350026668429, max(var)=270.9880129654529

   when dx=1e-9,  min(mean)=1.1362077297534983, max(mean)=1.5578893360531192, 

   min(var)=-3498.7493724270157, max(var)=3930.0980676451222

   

   When dx is smaller, the var we get is bigger. When dx is even smaller, the var we get contains negative values. Since we exclude the negative values in the next step, the price we get should be bigger.

   **preliminary solutions:**

   1. check the properties of cond_avgvar_mv_numeric, or the Laplace transform of integral variance, does it change a lot when dx changes?
   2. compare with the analytic value of mean and var 

2. In Case II, bias is large. The bias is [0.1906 0.2345 0.2568].

   **Location:** cannot find the problem directly. 

   **Guess:** 

   1. The problem may caused by the wrong var, just same as Case I, so we need to compare it with the analytic value of mean and var. 

      Then we need to modify the analytical mean and var, comparing the numerical value in Case I above.

   2. The problem may caused by the higher moments of the integral variance. Then it is need to introduce higher moments such as skewness, to decrease the bias.

      Before introducing the skewness,

      In Case I, the bias is -0.006369960972332722

      In Case II, the  bias is [0.1906 0.2345 0.2568]

      In Case III, the bias is [0.0188 0.0226 0.0266]

      In Case IV, the bias is [-5.563  -5.2922 -5.1399 -4.9083 -4.6686]

      In Case V, the bias is [-5.2457 -5.2457 -4.4213]

      In Case VI, the bias is [0.0106 0.0139 0.0176 0.0212 0.0242]

      

      After introducing the skewness,  In Case II

      max(skew) =115843.94832703802, min(skew)=-265187.0750326773

      but there are 5921 nan in avgvar, which is caused by the infinity of lambda. Here the number of inf is equal to 5921. This is caused by the 0 skew we may get. Set the skew=eps when it equals to 0. Then the answer is nan. This is caused by negative avgvar. There are 94 negative values. Set the negative avgvar=0, the bias is [-2.0902 -1.9316 -1.635 ].
      

      Then we further check the results in other cases:

      In Case I, the bias is -0.05406009189486538

      In Case II, the  bias is [-2.0902 -1.9316 -1.635 ]

      In Case III, the bias is [-1.0139 -0.958  -0.8664]

      In Case IV, the bias is [-5.563  -5.2922 -5.1399 -4.9083 -4.6686]

      In Case V,  the bias is [-5.2457 -5.2457 -4.4213]

      In Case VI, the bias is [5215.4543 5215.4389 5215.4239 5215.409  5215.3937]

      

      We can find that the bias is worse than before. Check further, 

      before introducing skew,max(avgvar)=0.2868717545699741, min(avgvar)=0.004794845986689878, mean(avgvar)=0.08137357127389411

      after introducing skew, max(avgvar)=1.4296163925154035, min(avgvar)=0, mean(avgvar)=0.08152858994639325

      we can see the shape of avgvar is changed, but the mean doesn't change.

      

      Then check further the different parameters' influences. Since in Case I, IV, V, the difference between with skew and without skew is small. We focus on Case II, III, VI

      In Case II

      for sigma

      * sigma=0.06, 

        Euler: [10.4557  7.4769  5.0312]
        IG: [10.5546  7.6205  5.1948]

      * sigma=0.12,

        Euler: [11.5568  8.6379  6.1876]
        IG: [11.5006  8.6327  6.2122]

      * sigma=0.18

        [12.1777  9.291   6.8434]
        [12.0254  9.1902  6.7743]

      * sigma=0.24

        [12.6145  9.7495  7.3054]
        [12.3745  9.5615  7.1509]

      * sigma=0.30

        [12.9861 10.1371  7.6963]
        [12.6379  9.8417  7.436 ]

      * sigma=0.50

        [14.0247 11.2096  8.7776]
        [13.225  10.4619  8.0659]

      * sigma=0.75

        [15.9406 13.1489 10.7231]
        [13.6719 10.9327  8.5451]

      * sigma=1

        [17.4751 14.698  12.2757]
        [13.9787 11.2553  8.8735]

        Here the bias decreases to negative when sigma increases. 

      for mr

      * mr = 22.84

        [10.4557  7.4769  5.0312]
        [10.5546  7.6205  5.1948]

      * mr = 20

        [10.2511  7.2502  4.7992]
        [10.3494  7.3885  4.9524]

      * mr=15

        [9.9756 6.935  4.4785]
        [9.9942 6.9867 4.5321]

      * mr=10

        [9.6887 6.6092 4.1532]
        [9.6157 6.5591 4.0953]

      * mr=5

        [9.3757 6.259  3.8112]
        [9.2054 6.0961 3.6238]

      * mr=2.5

        [9.2299 6.0954 3.6549]
        [9.0119 5.8747 3.4073]

      * mr=1.25

        [9.1533 6.0102 3.5741]

        [8.9003 5.7544 3.287 ]

      * mr=0.5

        [9.0808 5.9327 3.4996]

        [8.8487 5.6968 3.2344]

        Here is no obvious pattern.

      for theta

      * theta=0.218

        [10.4557  7.4769  5.0312]
        [10.5546  7.6205  5.1948]

      * theta=0.2

        [10.234   7.2423  4.7993]
        [10.3519  7.4028  4.9773]

      * theta=0.15

        [9.7169 6.6894 4.2584]
        [9.8045 6.8111 4.3881]

      * theta=0.1

        [9.1904 6.1287 3.7167]
        [9.2766 6.2378 3.8236]

      * theta=0.08

        [9.0071 5.9322 3.5297]
        [9.0753 6.0183 3.6095]

      * theta=0.05

        [8.7263 5.6318 3.246 ]
        [8.7808 5.6982 3.3014]

      * theta=0.01

        [8.3337 5.2143 2.8544]

        [8.4065 5.2948 2.9222]

      * theta=0.3

        [11.4788  8.5593  6.1103]
        [11.4715  8.6015  6.1808]

      * theta=0.4

        [12.9535 10.1035  7.6658]
        [12.5659  9.7651  7.358 ]

      * theta=0.5

        [14.4157 11.6311  9.2148]
        [13.6085 10.8667  8.4784]

      * theta=0.6

        [16.6438 13.9183 11.5271]
        [14.6098 11.9197  9.5524]

      * theta=0.7

        [18.6946 16.0231 13.6591]
        [15.5501 12.9074 10.5623]

        The bias decreases when theta getting smaller, increases when theta getting bigger.

      for vov

      * vov = 8.56

        [10.4557  7.4769  5.0312]
        [10.5546  7.6205  5.1948]

      * vov=7

        [10.8407  7.9214  5.5008]
        [11.0302  8.1421  5.7271]

      * vov=6

        [11.1135  8.2392  5.843 ]
        [11.3014  8.4491  6.059 ]

      * vov=5

        [11.4181  8.5954  6.2318]
        [11.5713  8.767   6.408 ]

      * vov=4

        [11.7036  8.9404  6.6187]
        [11.7923  9.0447  6.7283]

      * vov=3

        [11.9521  9.2563  6.9872]
        [12.0043  9.3197  7.0566]

      * vov=2

        [12.1426  9.5219  7.3157]
        [12.2252  9.6028  7.3948]

      * vov=1

        [12.2547  9.7144  7.5802]
        [18.6972 16.0956 13.7745]

      * vov=0.5

        [12.2762  9.777   7.6813]

        [1528.4925 1523.4925 1518.4925]

        When vov is near 4, the bias is small. When vov is less than 1, the bias becomes bigger.

      for rho

      * rho=-0.99

        [10.4557  7.4769  5.0312]
        [10.5546  7.6205  5.1948]

      * rho=-0.8

        [10.4497  7.5497  5.1818]
        [10.5312  7.6428  5.2679]

      * rho=-0.7

        [10.4493  7.5898  5.2626]
        [10.5168  7.6572  5.3153]

      * rho=-0.6

        [10.4487  7.6295  5.3427]
        [10.5004  7.6729  5.3677]

      * rho=-0.5

        [10.4465  7.6675  5.421 ]
        [10.4819  7.6895  5.424 ]

      * rho=-0.4

        [10.4465  7.6675  5.421 ]
        [10.4819  7.6895  5.424 ]

      * rho=-0.3

        [10.4335  7.7349  5.5687]
        [10.4387  7.7246  5.5452]

      * rho=-0.2

        [10.4211  7.7629  5.6367]
        [10.4142  7.7427  5.6084]

      * rho=-0.1

        [10.4041  7.7865  5.7003]
        [10.3877  7.7606  5.672 ]

        When rho is bigger than -0.3, the bias is small.

        

      **preliminary solutions:**

      1. Compare with the analytic value of mean and var.
      2. Introduce higher moments to make the approximation more explicit.
      3. Try other distributions to better approximating.

      

      

      

      

   Taylor expansion, or analytical derivative

   

   # 9.9-11.9 Task: Analytical Form of Derivatives of Laplace Transform

   Needed: Taking derivative w.r.t. of $a^*$ 
   
   ![image-20240919145245627](C:\Users\27261\AppData\Roaming\Typora\typora-user-images\image-20240919145245627.png)
   
   This is the information of bessel function：https://functions.wolfram.com/Bessel-TypeFunctions/BesselI/ 
   
   

