# A-B_Testing
It includes A/B test scenarios that can be performed between two or more groups in terms of ratio and average.〽

# AB Testing (Two-Sample Independent T-Test)

 1. Build Hypotheses
 2. Assumption Control : These methods allow us to make some ambitious generalizations from a high level; I made an improvement on my website, I am measuring its effects, I will make an interpretation according to these effects, I want to base this interpretation on a scientific basis, hypothesis testing will provide this process
   - 1. Assumption of Normality: That the distributions of the relevant groups are normal
   - 2. Homogeneity of Variance: The distributions of the variances of two groups are similar to each other
 3. Implementation of the Hypothesis
   - 1. If the assumptions are met, independent two sample t test (parametric test)
   - 2. If assumptions are not met, mannwhitneyu test (non-parametric test)
 4. Interpret results according to p-value

 Note
 - If normality is not ensured, direct non parametric test is applied. If variance homogeneity is not ensured, parametric test is applied, but the method is informed that variance homogeneity is not ensured. 
 - It may be useful to perform outlier analysis and correction before normality analysis.
## Application 1

# AB Testing (Two Sample Proportion Test)

## Application 2:

For more information visit my medium article -->  https://medium.com/@merveatasoy48/ab-testing-65b529266768

