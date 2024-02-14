

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################



############################
# Application 1 : Is There a Statistically Significant Difference Between the Mean Ages of People with and Without Diabetes?
############################

df = pd.read_csv(r"datasets\diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})
#outcome : 1 diabetes status
#There seems to be a difference according to the table, but let's base it on a scientific basis that emerged by chance


# 1. Set up hypotheses
# H0: M1 = M2
# There is no statistically significant difference between the mean age of those with and without diabetes
# H1: M1 != M2
# ....

# 2. Assumption Control

# Assumption of Normality (H0: Assumption of normal distribution is met.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9546, p-value = 0.0000
#Ho rejected

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.8012, p-value = 0.0000
#Ho rejected


# non-parametric as normality assumption is not met.
# non-parametric can also be thought of as a comparison of medians

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 92050.0000, p-value = 0.0000
#Ho rejected, the average age of people with and without diabetes is not equal.


######################################################
# AB Testing (Two Sample Proportion Test)
######################################################

# H0: p1 = p2
# There is no statistically significant difference between the conversion rate of the new design and the conversion rate of the old design.
# H1: p1 != p2
# ...

#1. ve 2. grubun başarı sayısı
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)
#(3.7857863233209255, 0.0001532232957772221)
# p value < 0.05 --> 0.0001532232957772221
# Ho rejected,

basari_sayisi / gozlem_sayilari
#array([0.3       , 0.22727273])


############################
# Application 2: Is there a Statistically Significant Difference between the Survival Rates of Women and Men?
############################

# H0: p1 = p2
# There is no Statistically Significant Difference between the Survival Rates of Women and Men
# H1: p1 != p2
# There are.. 

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()
#0.7420382165605095

df.loc[df["sex"] == "male", "survived"].mean()
#0.18890814558058924

#proportions_ztest expects the number of successes for the first argument and the number of observations for the second argument


female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()  
#233
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()
#109

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 16.2188, p-value = 0.0000
#H0 is rejected, male and female survival rates are not the same

