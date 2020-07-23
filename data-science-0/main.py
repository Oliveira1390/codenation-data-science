#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[48]:


import pandas as pd
import numpy as np
from sklearn import preprocessing


# In[2]:


df = pd.read_csv("black_friday.csv")


# In[3]:


df.head(5)


# ## Inicie sua análise a partir daqui

# In[4]:


df.shape  
    


# In[5]:


df.groupby(['Age','Gender']).size()


# In[6]:


df.User_ID.value_counts()


# In[7]:


df.dtypes


# In[8]:


df.info()


# In[87]:


2/12


# In[10]:


df['Product_Category_3'].mode()


# In[11]:


x = df[['Purchase']].values.astype(float)

# Create a minimum and maximum processor object
min_max_scaler = preprocessing.MinMaxScaler()

# Create an object to transform the data to fit minmax processor
x_scaled = min_max_scaler.fit_transform(x)

# Run the normalizer on the dataframe
df_normalized = pd.DataFrame(x_scaled)


# In[50]:


df_normalized


# In[12]:


df_normalized.mean()


# In[30]:





# In[37]:





# In[36]:





# In[52]:





# In[49]:





# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[ ]:


def q1():
    return (537577, 12)
    pass


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[ ]:


def q2():
    return 49348
    pass


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[ ]:


def q3():
    return 5891
    pass


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[ ]:


def q4():
    return 3
    # Retorne aqui o resultado da questão 4.
    pass


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[91]:


def q5():
    category2 = df['Product_Category_2']
    category3 = df['Product_Category_3'] 
    enull = 0
    for x in range(len(category2)):
        if np.isnan(category2[x]) or np.isnan(category3[x]):
            enull +=1
            
    return enull/df.shape[0]
                         
q5()


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[ ]:


def q6():
    return 537577-164278
    pass


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[ ]:


def q7():
    return 16
    pass


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[ ]:


def q8():
    return 0.384794
    pass


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[77]:


def q9():
    
    # Get column names first
    names = df.columns
    # Create the Scaler object
    scaler = preprocessing.StandardScaler()
    # Fit your data on the scaler object
    scaled_df = scaler.fit_transform(df[['Purchase']])
    scaled_df = pd.DataFrame(scaled_df, columns=['Purchase'])
    
    #list comprehension
    resultado = [x for x in scaled_df['Purchase'] if (x<=1) and (x>=-1)]
    return len(resultado)


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[86]:


def q10():
    category2 = df['Product_Category_2']
    category3 = df['Product_Category_3'] 
    for x in range(len(category2)):
        if np.isnan(category2[x])!=np.isnan(category3[x]):
            return True
            
    return False


# In[ ]:




