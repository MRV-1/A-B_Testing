######################################################
# Temel İstatistik Kavramları
######################################################

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


############################
# Sampling (Örnekleme)
############################
#AMAÇ : Popülasyon ve örneklem arasındaki ilişkinin veri dünyasına ne şekilde yansıdığını görmeye çalışmak

#hedeflenen lokasyondaki 10.000 kişinin yaş ortalamasına  erişilmek isteniyor
#örnek teorisi derki : 10.000 kişiyi temsil eden iyi bir  alt küme seç bu grup rastgele, yansız, olsun ana kitleyi iyi temsil etsin
#örnek teorisi 10.000 kişiyi gezmeden genelleme yapma durumu oluşturur

populasyon = np.random.randint(0, 80, 10000)
# burada insanların yaşları olduğunu varsayalım

populasyon.mean()

np.random.seed(115)    #aynı sonuçlar gelsin diye

orneklem = np.random.choice(a=populasyon, size=100)
orneklem.mean()

#örneklem daha az veri ile genellemeler yapabilmeyi sağlar
#sonuçlar iki kez çalıştırıldığında 1 br'lik fark oluşmuştur
#ilçedeki 10.000 kişiyi gezmek yerine örneklem bize bir değer oluşturmuştur, belirli bir yanılma payı ile oran budur şeklinde açıklanabilir
#örneklem bize daha az veri ile genellemeler yapabilmeyi sağlar, (zaman para iş gücü açısından kolaylık sağlar)


#örneklem sayısı artırıldığında  bu örneklem dağılımına ilişkin ortalama da popülasyona yakınsıyor olacaktır
#veri arttığında ya da örnek sayısı arttığında bir yakınsama söz konusudur


np.random.seed(10)
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10


############################
# Descriptive Statistics (Betimsel İstatistikler)
############################
#keşifçi veri analizi/betimsel istatistik/tanımlayıcı-açıklayıcı istatikler
#veri setini betimlemeye çalışma çabasıdır

df = sns.load_dataset("tips")
df.describe().T

#değişkenin değeri çarpık yani içerisinde aykırı değerler var ise bu değişkeni temsil etmek için medyan kullanılmalıdır
#değişken içerisinde aykırı değer olup olmadığını anlamak için --> ortalama ile medyan kıyaslandığında birbirine çok yakın değerler içeriyorsa
#değişkeni temsil etmek için kullanılacak olan istatistiğin medyan ya da ortalama olması farketmeyecektir
#ama ikisi arasında ciddi farklılık varsa bu durumda ortalama, medyan ve standart sapması şu şekildedir şeklinde bilgi vermek mantıklıdır


############################
# Confidence Intervals (Güven Aralıkları)
############################


# Tips Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("tips")
df.describe().T

df.head()
#tips veri setindeki total_bill değişkeninin güven aralığı nedir
#evet ben bu ortalama değere sahibim ama ben öyle bir bilgiye sahip olmak istiyorumki kötü senaryoda ne kazanırım, ki ona göre çalışanlarımın maaşlarını belirlemek istiyor olabilirim ya da ödeme zamanlarını planlamak istiyor olabilirim ya da iyi sernayoda ne kazanacağımı bilmek istiyor olabilirim
#dolayısıyla ortalama bir hesap bilgisi yeterli olmamaktadır


sms.DescrStatsW(df["total_bill"]).tconfint_mean()  #tconfint_mean kullanılarak güven aralığı hesabı gerçekleştiriliyor
# sonuç : (18.66333170435847, 20.908553541543164)
# mean : 19.785943
# mean elde edilen değerlerin aralığındadır
# elde edilen sonuçlar  %95 bu aralıkta %5(hata payı) bu aralığın dışında çıkacaktır
# buradan şu sonuç elde edildi : restorantın müşterilerinin ödediği hesap aralığı %95 güven ile .... değerleri arasındadır

sms.DescrStatsW(df["tip"]).tconfint_mean()




# Titanic Veri Setindeki Sayısal Değişkenler için Güven Aralığı Hesabı
df = sns.load_dataset("titanic")
df.describe().T
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
#(28.631790041821507, 30.766445252296133)
#ortalaması 29.69, medyan 28 anormal bir durum var gibi görünmüyor

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()
#(28.936831234567332, 35.471584702581936)
#ortalaması 32, medyan 14 bu durum içerisinde aykırılıklar barındırabilceği anlamına geliyor
#yaş değişkeninin ara lığı 30-28 = 2, fare 35-28 = 7 br
#güven aralığından ortalamanın üzerine standart sapma/kök n eklenir, çıkarılır
# standart sapma bölü kök n sapmayı standartlaştırmak yani sapmanın ortalamasını almak demektir
#standart sapma eğer yüksek olursa bu aralığın daha geniş olması beklenir



######################################################
# Correlation (Korelasyon)
######################################################


# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

#normalde ödenen hesap arttıkça bahşiş artmalı, burada dikkat edilmesi gereken durum yemeğin toplam fiyatına bahşişte dahil durumudur


df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")
plt.show(block=True)
#dağılımına bakıldığında ikisi arasında pozitif yönlü bir ilişki görünüyor, ilişkinin şiddeti hakkında bir yorum yapılamasada orta şiddetin üstünde bir ilişki var gibi duruyor
#peki bu nasıl değerlendirilecek corr ile

df["tip"].corr(df["total_bill"])
#iki dğişken arasındaki değişimi gözlemlemek istediğimizde kullanabileceğimiz bir methoddur
#0.5766 --> ödenen hesap miktarı arttıkça bahşişte artacaktır


######################################################
# AB Testing (Bağımsız İki Örneklem T Testi)
######################################################
#AB testi senaryoları uygulanacak olduğunda bu akışta gidilmelidir, bu akışın pekişmesi önemlidir.


# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü : Bu yöntemler bize üst perdeden bazı iddialı genellemeler yapma imkanı sağlar; web sitemde bir geliştirme yaptım bunun etkilerini ölçüyorum, bu etkilere göre bir yorumda bulunacağım bu yorumu bilimsel bir temele dayandırmak istiyorum, hipotez testi bu işlemi sağlayacaktır
#   - 1. Normallik Varsayımı : İlgili grupların dağılımlarının normal olması
#   - 2. Varyans Homojenliği : İki grubun varyanslarının dağılımlarının birbirine benzer olması
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.



############################
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İstatistiki Olarak Anlamlı bir Fark var mı?
############################
# ya da mesela haftaiçi ödenen hesap ile haftasonu ödenen hesap arasında fark var mı sorusuda araştırılabilir ?


df = sns.load_dataset("tips")
df.head()

df.groupby("smoker").agg({"total_bill": "mean"})

#bu durum rastgele ortaya çıkmış olabilir, bilimsel temele dayandıralım

############################
# 1. Hipotezi Kur
############################

# H0: M1 = M2
# H1: M1 != M2

############################
# 2. Varsayım Kontrolü
############################

# Normallik Varsayımı
# Varyans Homojenliği

############################
# Normallik Varsayımı
############################

# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır.

#Shapiro testi bir değişkenin dağılımının normal olup olmadığını test eder


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value < ise 0.05'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.


test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#iki grup içinde H0 reddedildi dolayısıyla iki grup içinde normal dağılım varsayımı sağlanmamaktadır.
#bu durumda non parametrik bir test kullanmamış gerekir

############################
# Varyans Homojenligi Varsayımı
############################

# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir

#levene; bana iki farklı grubu gönder ben bu iki farklı gruba göre sana varyans homojenliği varsayımının sağlanıp sağlanmadığını ifade edeyim.


test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# p-value = 0.0452 çıktığı için H0 reddedilir,  varyanslar homojen değildir

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 3 ve 4. Hipotezin Uygulanması
############################

# 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

############################
# 1.1 Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
############################

# ttest_ind eğer --> normallik varsayımı + varyans homojenliği sağlanıyorsa da beni kullan,
# normallik varsayımı sağlanıyor --> varyans homojenliği sağlanmıyorsa da beni kullan ama bu durumda  equal_var=False gir
test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"],
                              equal_var=True)

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 1.3384, p-value = 0.1820
# p-value > 0.05 olduğundan Ho reddedilemez.
# yani iki grup birbirine eşittir

# p-value < ise 0.05 'ten HO RED.
# p-value < değilse 0.05 H0 REDDEDILEMEZ.

############################
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
############################

# mannwhitneyu non parametrik medyan kıyaslama testidir

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 7531.5000, p-value = 0.3413

# p-value > 0.05 olduğundan Ho reddedilemez.
# yani iki grup birbirine eşittir

#ÖNEMLİ : Ho ya reddedilir, ya reddeedilmez. H1'i kabul etmek diye bir durum yoktur.



############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})
#tabloya göre fark var gibi duruyor ama bu fark şans eserimi ortaya çıktı bilimsel bir temele dayandıralım



# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)


# 2. Varsayımları İncele

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır


test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9848, p-value = 0.0071 Ho reddedilir

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9747, p-value = 0.0000 Ho reddedilir

#iki grup içinde varsayım sağlanmamaktadır
#varsayım sağlanmadığı için gidilecek yer non-parametreik testtir

# Varyans homojenliği
# H0: Varyanslar Homojendir
# H1: Varyanslar Homojen Değildir
#normalde ilk kısımdan sonra gidilecek yer non-parametrik testtir fakat, bu testide bir hatırlama amaçlı inceleyelim



test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.0013, p-value = 0.9712
# Ho reddedilemez, varyanslar homojendir anlamına gelir
#zaten non-parameetriğe düşmüştük, yine de baktık buna

# Varsayımlar sağlanmadığı için nonparametrik

test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 53212.5000, p-value = 0.0261
#Ho reddedilir

# 90 280 : elimizd böyle bir ortalama olduğunda bir test yapma ihtiyacı duymayız
# örn : bir ilaç geliştirilir bu ilaç bir gruba verilir ve diğer gruba verilmez v etkileri incelenir
# ilacın verildiği grupta bir etki görülür ama bu etki kayda değer mi değil mi bilinemez işte burada istatistiki olarak fark vardr ya da yoktur deme ihtiyacı vardır




############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv(r"C:\Users\MerveATASOY\Desktop\data_scientist_miuul\egitim_teorik_icerikler\Bolum_5_Measurement_Problems\datasets\diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})
#outcome : 1 diyabet olma durumu
#tabloya gör bir fark var gibi duruyor ama şans eserimi ortaya çıkmış bilimsel bir temele dayandıralım


# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.9546, p-value = 0.0000
#Ho reddedilir

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.8012, p-value = 0.0000
#Ho reddedilir


# Normallik varsayımı sağlanmadığı için nonparametrik.
# non parametrik medyanların kıyaslaması şeklinde de düşünülebilir

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 92050.0000, p-value = 0.0000
#Ho reddedilir, diyabet olanlar ve olmayanların yaş ortalamaları eşit değildir



###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

#online bir eğitim platformunda bir eğitmen grubu bir yargıya varmak istiyor; kursların büyük çoğunluğu izleyenler ile izlemeyenler arasında istatistiki olarak anlamlı  bir fark var mıdır


df = pd.read_csv(r"C:\Users\MerveATASOY\Desktop\data_scientist_miuul\egitim_teorik_icerikler\Bolum_5_Measurement_Problems\datasets\course_reviews.csv")
df.head()

df[(df["Progress"] > 75)]["Rating"].mean()
# 4.860491071428571

df[(df["Progress"] < 25)]["Rating"].mean()
# 4.7225029148853475

#farklılık var istatistiksel olarak anlamlı mı?

test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.3160, p-value = 0.0000
#Ho reddedilir, normallik varsayımı sağlanmadı

test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.5710, p-value = 0.0000
#Ho reddedilir, normallik varsayımı sağlanmadı
#non parametrik teste düştü


test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])


print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 661481.5000, p-value = 0.0000
#Ho reddedilir, iki grup eşit değildir

######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

#1. ve 2. grubun başarı sayısı
basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)
#(3.7857863233209255, 0.0001532232957772221)
# p value < 0.05 --> 0.0001532232957772221
# Ho reddedilir,

basari_sayisi / gozlem_sayilari
#array([0.3       , 0.22727273])

#gerçek hayatta bu oranları elde etmek zordur

############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()
#0.7420382165605095

df.loc[df["sex"] == "male", "survived"].mean()
#0.18890814558058924

#proportions_ztest ilk argümanına başarı sayısını, ikinci argümanına gözlem sayısını bekler


female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()  #hayatta kalmalar sum edilirse başarı oranı bulunur
#233
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()
#109

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 16.2188, p-value = 0.0000
#H0 reddedilir,  erkek ve kadınların hayatta  kalma oranları aynı değildir

######################################################
# ANOVA (Analysis of Variance)
######################################################

# İkiden fazla grup ortalamasını karşılaştırmak için kullanılır.
# Haftanın günleri arasında farklılık olup olmadığı karşılaştırılmak isteniyor.

df = sns.load_dataset("tips")
df.head()

df.groupby("day")["total_bill"].mean()
"""
day
Thur   17.68274
Fri    17.15158
Sat    20.44138
Sun    21.41000
"""
#hafta içi vee haftasonu kendi içinde yakın ama hafta içi ve sonu arasında ufak bir farklılık var gibi duruyor

# 1. Hipotezleri kur

# HO: m1 = m2 = m3 = m4
# Grup ortalamaları arasında fark yoktur.

# H1: .. fark vardır

# 2. Varsayım kontrolü

# Normallik varsayımı
# Varyans homojenliği varsayımı

# Varsayım sağlanıyorsa one way anova
# Varsayım sağlanmıyorsa kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır.


#öyle bir işlem yapmalıyımi veri setindeki day değişkenini günlere göre filtreleyeyim sonrasında buna shapiro uygulayayım

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, 'p-value: %.4f' % pvalue)

"""
Sun p-value: 0.0036
Sat p-value: 0.0000
Thur p-value: 0.0000
Fri p-value: 0.0409
"""
#hepsi için p değeri 0.05'ten küçük olduğundan dolayı Ho hipotezi reddedilir
#Dolayısıyla hiç biri için normallik varsayımı sağlanmamaktadır

#df içerisindeki unique day değerlerinin içerisinde gez, bunu listeye çevir
#kategorik değişkenin sınıfları iteratif bir nesneye çevrildi



# H0: Varyans homojenliği varsayımı sağlanmaktadır.

test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
#Test Stat = 0.6654, p-value = 0.5741
#Ho reddedilemez, varyans homojenliği varsayımı sağlanmaktadır.
#Ama normallik varsayımı testinden kalındığı için non parametriğe geçildi

#iki grup arasında fark var mı yokmuyu birden fazla grup içerisinde bunları grup grup çaprazlayarak test etmek ile ikiden fazla grup arasında fark var mı yokmuyu tek seferde test etmek aynı şey değildir
#Anova gibi yöntemler kullanıldığında grup içi ve gruplar arası değişkenler göz önünde bulundurulur bu bir parametredir, dolayısıyla hesaplanacak olan test istatistiği ve p-value değerine göre yapılacak olan yorum ile gruplar kendi içinde birebir karşılaştırıldığında ortaya çıkabilecek yorumlar farklı olacaktır

#

# 3. Hipotez testi ve p-value yorumu

# Hiç biri sağlamıyor.
df.groupby("day").agg({"total_bill": ["mean", "median"]})


# HO: Grup ortalamaları arasında ist ol anl fark yoktur

# parametrik anova testi:
f_oneway(df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Sun", "total_bill"])
# kullanımı göstermek için kullandı


# Nonparametrik anova testi:
kruskal(df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Sun", "total_bill"])

# KruskalResult(statistic=10.403076391437086, pvalue=0.01543300820104127)
#Ho reddedilir, gruplar arasında istatistiki olarak anlamlı bir fark vardır

#Fark var ama hangi gruptan kaynaklanıyor ?
#bunun için kullanılabilecek olan farklı kütüphaneler ve yöntemler var statsmodel kullanacağız


from statsmodels.stats.multicomp import MultiComparison
comparison = MultiComparison(df['total_bill'], df['day'])
tukey = comparison.tukeyhsd(0.05)
print(tukey.summary())

# Cuma ile Cumartesinin ortalamaları arasındaki fark 3.28, düzeltilmiş p-value değeri 0.45 Ho reddedilemedi dolayısıyla bu ikisi arasında fark yoktur
# hiç biri arasında istatistiki olarak fark yok
# ikili karşılaştırma yapıldığında istatistiksel olarak anlamlı bir fark bulunamadı, toplu bakıldığında farklıydı
# tek başına bütün gruba bakıldığında bir farklılık bulundu ama bu farkı ikili karşılaştırmalarda göremedim, dolayısıyla fark yok muamelesi yapmak tercih edilebilir
# bütün  gruba anova açısından f testiyle bakmakla grup içi ve gruplar arası değişkenliği değerlendirmek birbirinden farklıdır
# alfa değeri değiştirilebilir, ön tanımlı 0.05'tir
# Burada bir fark bulunamadıysa önceki adıma geri dönülüp fark yokmuş gibi kabul edilebilir, çünkü bu aşamada hangi grubun diğerlerine göre baskınlığı varsa yansımasını bekleriz ama buraya yansımamış

