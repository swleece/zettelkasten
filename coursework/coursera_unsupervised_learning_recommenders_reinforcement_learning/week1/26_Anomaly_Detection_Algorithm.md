<26>

## Title
Predicting anomalies with gaussian distributions

## Description
1. choose n features xi that may be indicative of anomalous examples
2. fit parameters µ , sigma^2
3. given new example, compute p(x)
4. **anomalous** if p(x) < epsilon
5. choose a reasonable epsilon based on performance and application



## Additional Notes
Gaussian distribution (can be vectorized):
	µj = simple average of xj
	sigma^2 = average of squared difference
 ![[gaussian_distribution.png]]
Density estimation:
p(vector x) = p(x1; µ1, sigma1) * (...n)
		= product from j=1 to n of p(xj; µj, sigmaj^2)
		= product from j=1 to n of (1 / pi*root(2)**)
![[density_estimation.png]]

In practice, useful to have a small amount of labelled data where some examples are anomalous.
Common to only use *normal* examples in training, then half of your anomalous examples in cross validation set, half in test set.
E.g. if your data 10000 normal engines, 20 flawed, you might split that into:
- Training: 6000 good engines
- CV and Test sets each: 2000 good, 10 anomalous
Alternatively, you may remove the Test set entirely and just tune with the CV set including all of the anomalous examples.




## Linked Cards
{{ direct link to another card }}

## Tags
[[Machine Learning]] 