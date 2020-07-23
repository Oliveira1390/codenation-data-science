#!/usr/bin/env python
# coding: utf-8

# # Desafio 3
# 
# Neste desafio, iremos praticar nossos conhecimentos sobre distribuições de probabilidade. Para isso,
# dividiremos este desafio em duas partes:
#     
# 1. A primeira parte contará com 3 questões sobre um *data set* artificial com dados de uma amostra normal e
#     uma binomial.
# 2. A segunda parte será sobre a análise da distribuição de uma variável do _data set_ [Pulsar Star](https://archive.ics.uci.edu/ml/datasets/HTRU2), contendo 2 questões.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Setup_ geral

# In[112]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sct
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn import preprocessing
import scipy


# In[113]:


#%matplotlib inline

#from IPython.core.pylabtools import figsize

#figsize(12, 8)

#sns.set()


# ## Parte 1

# ### _Setup_ da parte 1

# In[114]:


np.random.seed(42)
    
df = pd.DataFrame({"normal": sct.norm.rvs(20, 4, size=10000),
                     "binomial": sct.binom.rvs(100, 0.2, size=10000)})

df.head(5)


# ## Inicie sua análise a partir da parte 1 a partir daqui

# In[115]:


# Sua análise da parte 1 começa aqui.

df.shape,df.describe()


# In[116]:


#Questao 1
q1_norm = df['normal'].quantile(.25)
q2_norm = df['normal'].quantile(.50)
q3_norm = df['normal'].quantile(.75)

q1_norm,q2_norm,q3_norm


# In[117]:


q1_binom = df['binomial'].quantile(.25)
q2_binom = df['binomial'].quantile(.50)
q3_binom = df['binomial'].quantile(.75)

q1_binom,q2_binom,q3_binom


# In[118]:


q1_norm - q1_binom, q2_norm - q2_binom, q3_norm - q3_binom


# In[119]:


q1_dif = np.round(q1_norm - q1_binom,3)
q2_dif = np.round(q2_norm - q2_binom,3)
q3_dif = np.round(q3_norm - q3_binom,3)

q1_dif, q2_dif, q3_dif


# In[120]:


#Questao 2
df_array = np.array(df['normal'])
normalized_df = preprocessing.normalize([df_array])
df_array.mean(),df_array.std(),normalized_df,normalized_df.mean(),normalized_df.std()


# In[121]:


sns.distplot(df_array);


# In[122]:


sns.distplot(normalized_df);


# In[123]:


ecdf = ECDF(df['normal'])
    
normal_std = df['normal'].std()
normal_mean = df['normal'].mean()
    
upper_1 = ecdf(normal_mean+normal_std)
lower_1 = ecdf(normal_mean-normal_std)
    
resultado = float((upper_1-lower_1).round(3))
resultado
    


# In[124]:


#Questao 3

m_binom = df['binomial'].mean()
v_binom = df['binomial'].var()
m_norm = df['normal'].mean()
v_norm = df['normal'].var()

r1 = float((m_binom - m_norm).round(3))
r2 = float((v_binom - v_norm).round(3))

r1,r2


# ## Questão 1
# 
# Qual a diferença entre os quartis (Q1, Q2 e Q3) das variáveis `normal` e `binomial` de `dataframe`? Responda como uma tupla de três elementos arredondados para três casas decimais.
# 
# Em outra palavras, sejam `q1_norm`, `q2_norm` e `q3_norm` os quantis da variável `normal` e `q1_binom`, `q2_binom` e `q3_binom` os quantis da variável `binom`, qual a diferença `(q1_norm - q1 binom, q2_norm - q2_binom, q3_norm - q3_binom)`?

# In[125]:


def q1():
    q1_norm = df['normal'].quantile(.25)
    q2_norm = df['normal'].quantile(.50)
    q3_norm = df['normal'].quantile(.75)
    
    q1_binom = df['binomial'].quantile(.25)
    q2_binom = df['binomial'].quantile(.50)
    q3_binom = df['binomial'].quantile(.75)
    
    q1_dif = np.round(q1_norm - q1_binom,3)
    q2_dif = np.round(q2_norm - q2_binom,3)
    q3_dif = np.round(q3_norm - q3_binom,3)

    
    return (q1_dif, q2_dif, q3_dif)
    


# Para refletir:
# 
# * Você esperava valores dessa magnitude?
# 
# * Você é capaz de explicar como distribuições aparentemente tão diferentes (discreta e contínua, por exemplo) conseguem dar esses valores?

# ## Questão 2
# 
# Considere o intervalo $[\bar{x} - s, \bar{x} + s]$, onde $\bar{x}$ é a média amostral e $s$ é o desvio padrão. Qual a probabilidade nesse intervalo, calculada pela função de distribuição acumulada empírica (CDF empírica) da variável `normal`? Responda como uma único escalar arredondado para três casas decimais.

# In[126]:


def q2():
   
    ecdf = ECDF(df['normal'])
    
    normal_std = df['normal'].std()
    normal_mean = df['normal'].mean()
    
    upper_1 = ecdf(normal_mean+normal_std)
    lower_1 = ecdf(normal_mean-normal_std)
    
    resultado = float((upper_1-lower_1).round(3))
    return resultado
    


# Para refletir:
# 
# * Esse valor se aproxima do esperado teórico?
# * Experimente também para os intervalos $[\bar{x} - 2s, \bar{x} + 2s]$ e $[\bar{x} - 3s, \bar{x} + 3s]$.

# ## Questão 3
# 
# Qual é a diferença entre as médias e as variâncias das variáveis `binomial` e `normal`? Responda como uma tupla de dois elementos arredondados para três casas decimais.
# 
# Em outras palavras, sejam `m_binom` e `v_binom` a média e a variância da variável `binomial`, e `m_norm` e `v_norm` a média e a variância da variável `normal`. Quais as diferenças `(m_binom - m_norm, v_binom - v_norm)`?

# In[127]:


def q3():
    
    m_binom = df['binomial'].mean()
    v_binom = df['binomial'].var()
    m_norm = df['normal'].mean()
    v_norm = df['normal'].var()
    
    r1 = float((m_binom - m_norm).round(3))
    r2 = float((v_binom - v_norm).round(3))
    
    return r1,r2
    


# Para refletir:
# 
# * Você esperava valore dessa magnitude?
# * Qual o efeito de aumentar ou diminuir $n$ (atualmente 100) na distribuição da variável `binomial`?

# ## Parte 2

# ### _Setup_ da parte 2

# In[128]:


stars = pd.read_csv("pulsar_stars.csv")

stars.rename({old_name: new_name
              for (old_name, new_name)
              in zip(stars.columns,
                     ["mean_profile", "sd_profile", "kurt_profile", "skew_profile", "mean_curve", "sd_curve", "kurt_curve", "skew_curve", "target"])
             },
             axis=1, inplace=True)

stars.loc[:, "target"] = stars.target.astype(bool)


# ## Inicie sua análise da parte 2 a partir daqui

# In[129]:


# Sua análise da parte 2 começa aqui.
stars.shape,stars.head()


# In[130]:


false_pulsar_mean_profile = stars[stars['target']==0]
false_pulsar_mean_profile.shape,false_pulsar_mean_profile.head(5)
s1 = false_pulsar_mean_profile['mean_profile']
sns.distplot(s1)


# In[131]:


z = (s1 - s1.mean()) / s1.std()
z.mean(),z.std()


# In[132]:


sns.distplot(z)


# In[139]:


ecdf = ECDF(z)
sct.norm.ppf([0.8, 0.90, 0.95])
quantis = sct.norm.ppf([0.8, 0.90, 0.95])
tuple(ecdf(quantis).round(3))


# In[144]:


false_pulsar_mean_profile = stars[stars['target']==0]
s1 = false_pulsar_mean_profile['mean_profile']
z = (s1 - s1.mean()) / s1.std()
tuple((z.quantile((0.25,0.5,0.75)) - sct.norm.ppf([0.25, 0.5, 0.75])).round(3))


# ## Questão 4
# 
# Considerando a variável `mean_profile` de `stars`:
# 
# 1. Filtre apenas os valores de `mean_profile` onde `target == 0` (ou seja, onde a estrela não é um pulsar).
# 2. Padronize a variável `mean_profile` filtrada anteriormente para ter média 0 e variância 1.
# 
# Chamaremos a variável resultante de `false_pulsar_mean_profile_standardized`.
# 
# Encontre os quantis teóricos para uma distribuição normal de média 0 e variância 1 para 0.80, 0.90 e 0.95 através da função `norm.ppf()` disponível em `scipy.stats`.
# 
# Quais as probabilidade associadas a esses quantis utilizando a CDF empírica da variável `false_pulsar_mean_profile_standardized`? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[136]:


def q4():
    
    false_pulsar_mean_profile = stars[stars['target']==0]
    false_pulsar_mean_profile.shape,false_pulsar_mean_profile.head(5)
    s1 = false_pulsar_mean_profile['mean_profile']
    
    z = (s1 - s1.mean()) / s1.std()
    z.mean(),z.std()
    
    ecdf = ECDF(z)

    quantis = sct.norm.ppf([0.8, 0.90, 0.95])
    
    return tuple(ecdf(quantis).round(3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?

# ## Questão 5
# 
# Qual a diferença entre os quantis Q1, Q2 e Q3 de `false_pulsar_mean_profile_standardized` e os mesmos quantis teóricos de uma distribuição normal de média 0 e variância 1? Responda como uma tupla de três elementos arredondados para três casas decimais.

# In[135]:


def q5():
    #remover false
    false_pulsar_mean_profile = stars[stars['target']==0]
    #tirar apenas mean_profile
    s1 = false_pulsar_mean_profile['mean_profile']
    #padronizando
    z = (s1 - s1.mean()) / s1.std()
    
    return tuple((z.quantile((0.25,0.5,0.75)) - sct.norm.ppf([0.25, 0.5, 0.75])).round(3))


# Para refletir:
# 
# * Os valores encontrados fazem sentido?
# * O que isso pode dizer sobre a distribuição da variável `false_pulsar_mean_profile_standardized`?
# * Curiosidade: alguns testes de hipóteses sobre normalidade dos dados utilizam essa mesma abordagem.
