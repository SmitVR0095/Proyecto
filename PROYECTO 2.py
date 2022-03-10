#!/usr/bin/env python
# coding: utf-8

# <center><img src="https://media-exp1.licdn.com/dms/image/C4E22AQEbIXZiRVkJPQ/feedshare-shrink_2048_1536/0?e=1598486400&v=beta&t=DSWeH9gVxZecLFd10nv-rGomMSJEJP7D2pXl60V1-9A" width="1000"></center>

# # PROYECTO 2
# ## Autor: Smit Jonatan Villafranca Romero
# ### INDICACIONES GENERALES:
# **Link de la data :** https://www.dropbox.com/s/tr8jf1dqlv4hsl8/TRAIN_FUGA.csv?dl=0
# 

# Librerias principales:

# In[1]:


import os            #Para direcionar la ruta de trabajo
import pandas as pd  #Para archivos csv, excel, spss, stata
import numpy as np   #Para trabajar con matrices/arrays
import matplotlib.pyplot as plt
import seaborn as sns
import math as math
import warnings
from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')


# ## CASO 1:
# Teniendo en cuenta la base de datos TRAIN_FUGA.csv que corresponde a los datos de entrenamiento de una data que analiza la fuga de clientes de una entidad bancaria realizar las siguientes tareas:
# 
# 1.	Análisis exploratorio de las variables: medidas y visualización. (5 puntos)
# 2.	Realizar un análisis exploratorio sobre presencia de outliers. (2 puntos)
# 3.	Realizar una discretización de las variables : INGRESO_BRUTO_M1 y EDAD teniendo en cuenta al menos dos técnicas de discretización no supervisada y agregar las variables discretizadas a nuestro conjunto de datos original (2 puntos)
# 4.	Aplicar dos técnicas de balanceo de datos a nuestra variable TARGET (objetivo) y agregarlas a nuestro conjunto de datos original. Use los parámetros vistos en clase. (3 puntos)
# 
# Nota: revisar el diccionario de variables del caso.
# 

# ## 1. Análisis exploratorio de las variables: medidas y visualización.

# ### 1.1. EXPLORACIÓN

# ### 1.1.1.  ANALIZANDO LOS DATOS

# In[2]:


os.chdir("E:\PYTHOM\MODULO 1\EXAMEN FINAL-MODULO1")#direccionando la ruta
archivo_csv="TRAIN_FUGA.csv"
df=pd.read_csv(archivo_csv,sep=",", encoding="ISO-8859-1")
df.head(50)


# #### Elimando la columna Unnamed: 0 debido a que esa columna no pertenece al DataFrame

# In[3]:


data_resultante=df.drop(["Unnamed: 0"], axis=1)
data_resultante.head(50)


# #### Información del DataFrame

# In[4]:


data_resultante.info()


# En la informacion de nuestra data podemos observar que hay datos faltantes

# #### Dimensión de nuestra data

# In[5]:


data_resultante.shape


# #### Veamos en que columnas se encuentran valores nulos(NAN)

# In[6]:


data_resultante.isnull().any()


# #### Ahora veamos cual es el porcentaje de esos valores nulos

# In[7]:


data_resultante.isnull().sum()/len(df)*100


# Podemos apreciar que hay datos que sobre pasan mas del 30% de los faltos faltantes por lo tanto procedemos a eliminar esas columnas ya que son perjudiciales para nuestro análisis

# #### Eliminando las columnas que poseen mas del 30% de los datos faltantes

# In[8]:


lista=["CODMES","TARGET_MODEL2","EDAD","SEXO","DEPARTAMENTO","INGRESO_BRUTO_M1","FLG_CLIENTE","SEGMENTO","FLG_ADEL_SUELDO_M1",
       "FREC_AGENTE","FREC_KIOSKO","FREC_BPI_TD","FREC_MON_TD","PROM_CTD_TRX_6M","ANT_CLIENTE","CTD_RECLAMOS_M1"]
data_resultante_2=data_resultante[lista]
data_resultante_2.head(50)


# In[9]:


data_resultante_2.info()


# In[10]:


data_resultante_2.isnull().sum()/len(data_resultante_2)*100


# #### Descripción de la data

# In[11]:


data_resultante_2.describe()


# ### 1.1.2. IMPUTACIÓN SIMPLE

# Las variables que haremos por imputación simple serán **DEPARTAMENTO(3.425419%)** y **ANT_CLIENTE(0.117207%)** debido a que tienes un bajo porcentaje de NaN

# #### Llamamos a nuestra DataFrame 

# In[12]:


data_resultante_2.head(50)


# #### Eliminamos la variable Ingreso Bruto 

# Debido a la variable **INGRESO_BRUTO_M1** tiene un porcentaje de 23.778437% de NaN, entonces no se puede aplicar una **IMPUTACIÓN SIMPLE** sino una **IMPUTACIÓN SOFISTICADA**. Por ello eliminamos por el momento esa variable y lo guardamos en un objeto para luego concatenarlo con las otras variables que han sido completadas

# In[13]:


X=data_resultante_2.drop(["INGRESO_BRUTO_M1"],axis=1)
X.head()


# #### Guardando la variable INGRESO_BRUTO_M1 

# In[14]:


y=data_resultante_2["INGRESO_BRUTO_M1"]
y.head(50)


# #### Sacando las columnas de mi objeto X (data sin la variable INGRESO_BRUTO_M1)

# In[15]:


columnas=X.columns.to_list()
columnas


# In[16]:


X.dtypes


# #### Agrupando según si tipo de variable

# In[17]:


X.columns.to_series().groupby(X.dtypes).size()


# In[18]:


tipos = X.columns.to_series().groupby(X.dtypes).groups
tipos


# Como se observa en la descripcion del DataFrame la columna del tipo entero no presenta ningún valor NaN, entonces solo completaremos las columnas del tipo categorico y flotante

# #### Armando las listas de las columnas categóricas y numéricas  (flotante)

# In[19]:


#Armando lista de columnas categóricas
#creamos la columna categórica
col_categorica = tipos[np.dtype('object')].to_list()
print("La cantidad de columnas con datos categóricos son: ",len(col_categorica))  
print("lista:\n",col_categorica)

#Armando lista de columnas numéricas de tipo flotante
col_flotante = tipos[np.dtype('float64')].to_list()
print("La cantidad de columnas con datos flotantes son: ",len(col_flotante)) 
print("lista:\n",col_flotante)


# #### Completanto los valores faltantes

# In[20]:


#Completando los valores faltantes para variables categóricas.
for cat in col_categorica:
     #Vamos a guardar la moda en el objeto moda y se utilizara el metodo .mode()
    moda=X[cat].mode()[0]
    #Vamos rellenar a los elementos vacios por la moda para eso utilizaremos el 
    #método .fillna(el valor que quieres almacenar)
    X[cat]=X[cat].fillna(moda)


# In[21]:


#Completando los valores faltantes para variables categóricas.
for flot in col_flotante:
     #Vamos a guardar la moda en el objeto moda y se utilizara el metodo .mode()
    mediana=X[flot].median()
    #Vamos rellenar a los elementos vacios por la moda para eso utilizaremos el 
    #método .fillna(el valor que quieres almacenar)
    X[flot]=X[flot].fillna(mediana)


# #### Verificamos si se completo 

# In[22]:


X.isnull().sum()/len(X)*100


# In[23]:


#Verificación de la data limpia de NAs
X.isnull().any().any()


# #### Mostramos la Data Frame resultante para las varibles que tuvieron un bajo porcentaje de NaN

# In[24]:


X.head(50)


# #### Concatenamos las Datas Frame (X e y)

# In[25]:


data_semifinal=pd.concat([X,y],axis=1)
data_semifinal.head(50)


# #### Veamos los porcentajes de los datos faltantes

# In[26]:


data_semifinal.isnull().sum()/len(data_semifinal)*100


# In[27]:


data_semifinal.describe()


# ### 1.1.3. IMPUTACIÓN POR MÉTODO SOFISTICADO

# In[28]:


from sklearn.linear_model import LinearRegression #Para imputación por regresión.


# In[29]:


data_semifinal.info()


# #### Vizualización de las relaciones entre variables cuantitativas

# In[30]:


with plt.style.context('dark_background'):
    sns.pairplot(data_semifinal)


# #### Hallando la correlación de las variables cuantitativas más relacionadas

# In[31]:


with plt.style.context('dark_background'):
    data_semifinal.plot.scatter(x="PROM_CTD_TRX_6M",
                          y="INGRESO_BRUTO_M1",
                         alpha=0.5)
plt.title("RELACIÓN ENTRE PROM_CTD_TRX_6M Y INGRESO BRUTO",weight='bold',color='red')
plt.show()


# In[32]:


P_C_T=data_semifinal["PROM_CTD_TRX_6M"]
ingreso=data_semifinal["INGRESO_BRUTO_M1"]
# PIB.cov(desempleo)
print("LA CORRELACION ES:",P_C_T.corr(ingreso)*100,"%")
print("LA COVARIANZA ES:",P_C_T.cov(ingreso))


# In[33]:


with plt.style.context('dark_background'):
    data_semifinal.plot.scatter(x="EDAD",
                          y="INGRESO_BRUTO_M1",
                         alpha=0.5)
plt.title("RELACIÓN ENTRE EDAD Y INGRESO BRUTO",weight='bold',color='red') 
plt.show()


# In[34]:


EDAD=data_semifinal["EDAD"]
ingreso=data_semifinal["INGRESO_BRUTO_M1"]
# PIB.cov(desempleo)
print("LA CORRELACION ES:",EDAD.corr(ingreso)*100,"%")
print("LA COVARIANZA ES:",EDAD.cov(ingreso))


# In[35]:


with plt.style.context('dark_background'):
    data_semifinal.plot.scatter(x="FREC_BPI_TD",
                          y="INGRESO_BRUTO_M1",
                         alpha=0.5)
plt.title("RELACIÓN ENTRE FREC_BPI_TD Y INGRESO BRUTO",weight='bold',color='red') 
plt.show()


# In[36]:


F_B_T=data_semifinal["FREC_BPI_TD"]
ingreso=data_semifinal["INGRESO_BRUTO_M1"]
# PIB.cov(desempleo)
print("LA CORRELACION ES:",F_B_T.corr(ingreso)*100,"%")
print("LA COVARIANZA ES:",F_B_T.cov(ingreso))


# In[37]:


with plt.style.context('dark_background'):
    data_semifinal.plot.scatter(x="ANT_CLIENTE",
                          y="INGRESO_BRUTO_M1",
                            alpha=0.5)
plt.title("RELACIÓN ENTRE ANT_CLIENTE Y INGRESO BRUTO",weight='bold',color='red')                     
plt.show()


# In[38]:


CLIENTE=data_semifinal["ANT_CLIENTE"]
ingreso=data_semifinal["INGRESO_BRUTO_M1"]
# PIB.cov(desempleo)
print("LA CORRELACION ES:",CLIENTE.corr(ingreso)*100,"%")
print("LA COVARIANZA ES:",CLIENTE.cov(ingreso))


# In[39]:


with plt.style.context('dark_background'):
    data_semifinal.plot.scatter(x="CTD_RECLAMOS_M1",
                          y="INGRESO_BRUTO_M1",
                         alpha=0.5)
plt.title("RELACIÓN ENTRE CTD_RECLAMOS_M1 Y INGRESO BRUTO",weight='bold',color='red')  
plt.show()


# In[40]:


RECLAMOS=data_semifinal["CTD_RECLAMOS_M1"]
ingreso=data_semifinal["INGRESO_BRUTO_M1"]
# PIB.cov(desempleo)
print("LA CORRELACION ES:",RECLAMOS.corr(ingreso)*100,"%")
print("LA COVARIANZA ES:",RECLAMOS.cov(ingreso))


# Llegamos a la conclusión que las variables mas relacionadas con la variable **INGRESO_BRUTO_M1** son:

# 1. PROM_CTD_TRX_6M(29.629876433451624 %)
# 2. FREC_BPI_TD(20.08935852242798 %)

# ### REGRESICIÓN LINEAL

# #### Mostrando los valores nulos

# In[41]:


nulos=pd.isna(data_semifinal.loc[:,"INGRESO_BRUTO_M1"])
nulos


# #### Almacenamos en el objeto "data_nueva_nulos" todos los datos nulos y en el objeto "data_nueva_completos" los datos completos

# In[42]:


data_semifinal_nulos=data_semifinal.loc[nulos]
data_semifinal_completos=data_semifinal.loc[~nulos]
data_semifinal_completos.info()


# In[43]:


xtrain=data_semifinal_completos[["PROM_CTD_TRX_6M","FREC_BPI_TD"]]
ytrain=data_semifinal_completos[["INGRESO_BRUTO_M1"]]

xtest=data_semifinal_nulos[["PROM_CTD_TRX_6M","FREC_BPI_TD"]]


# #### Crear un objeto de clase LinearRegression

# In[44]:


regre=LinearRegression()
type(regre)


# #### Aprendo del subconjunto de completos

# In[45]:


regre.fit(xtrain,ytrain)


# In[46]:


regre.score(xtrain,ytrain)


# #### obteniendo los datos faltantes

# In[47]:


ypredicho=regre.predict(xtest)
ypredicho=np.round(ypredicho,1)
ypredicho


# #### Mostramos los indeces donde los valores son nulos

# In[48]:


data_semifinal_nulos.index


# #### Incorporando los valores imputados al DF original

# In[49]:


data_semifinal.loc[data_semifinal_nulos.index,"INGRESO_BRUTO_M1"]=ypredicho
data_semifinal.head()


# #### Porcentaje de NAs por Columnas

# In[50]:


data_semifinal.isnull().sum()*100/len(data_semifinal)


# In[51]:


data_final=data_semifinal


# #### Guardando la data Imputada 

# In[52]:


data_final.to_csv("TRAIN_FUGA_COMPLETO.csv", index=False)
Image(filename='E:\PYTHOM\MODULO 1\EXAMEN FINAL-MODULO1/data_completa.png', width=900)


# ### 1.2. Visualización

# In[53]:


dat=pd.read_csv("TRAIN_FUGA_COMPLETO.csv",sep=",", encoding="ISO-8859-1")
dat.head(50)


# In[54]:


dat.info()


# ### 1.2.1. GRAFICAS DE SECTORES

# **Graficamos**

# In[55]:


desfase_1 = (0.04 ,0.04)
colors  = ("dodgerblue","salmon", "palevioletred", 
           "steelblue", "seagreen", "plum", 
           "blue", "indigo", "beige", "yellow")


# In[56]:


SEXO_frec=dat.groupby('SEXO').SEXO.count() 
SEXO_frec


# In[57]:


nombres_genero = ["FEMENINO","MASCULINO"]
with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(5,4))
    plt.pie(SEXO_frec,labels=nombres_genero,colors=colors,startangle=-60,explode=desfase_1,autopct='%1.2f%%')
plt.title("PORCENTAJE DEL GÉNERO DEL CLIENTE",weight='bold', size=14, loc='center',color="red")
plt.show()


# In[58]:


TARGET_MODEL2_freq =dat.groupby('TARGET_MODEL2').TARGET_MODEL2.count() 
TARGET_MODEL2_freq 


# In[59]:


nombre_cliente = ["NO FUGA","FUGA"]
TARGET_MODEL2_freq =dat.groupby('TARGET_MODEL2').TARGET_MODEL2.count() 
with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(5,4))
    plt.pie(TARGET_MODEL2_freq,labels=nombre_cliente ,colors=colors,startangle=10,explode=desfase_1,autopct='%1.2f%%')
plt.title("CALIFICACIÓN DEL CLIENTE",weight='bold', size=14, loc='center',color="red")
plt.show()


# In[60]:


FLG_CLIENTE_freq=dat.groupby('FLG_CLIENTE').FLG_CLIENTE.count()
FLG_CLIENTE_freq


# In[61]:


nombre_cliente_1= ["CLIENTE","NO CLIENTE"]
with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(5,4))
    plt.pie(FLG_CLIENTE_freq,labels=nombre_cliente_1,colors=colors,startangle=45,explode=desfase_1,autopct='%1.2f%%')
plt.title("PORCENTAJE ENTRE CLIENTE Y NO CLIENTE",weight='bold', size=14, loc='center',color="red")
plt.show()


# In[62]:


desfase = (0.1 ,0.1)


# In[63]:


fig= plt.figure(figsize=(30,10))
plt.subplot2grid((2,3),(0,0))
data_inicial = dat.groupby('FLG_CLIENTE').FLG_CLIENTE.count() 
nombres = ["Cliente","No Cliente"]
plt.pie(data_inicial, labels=nombres, autopct="%0.2f %%", colors=colors,
        explode=desfase,radius=5,pctdistance=0.6,rotatelabels=0,startangle=45)
plt.title("FLG_CLIENTE",fontsize=16, weight="bold")
plt.axis("equal")
#------------------------------------------------------------------------
plt.subplot2grid((2,3),(0,1))
data_final = data_resultante.groupby('FLG_CLIENTE').FLG_CLIENTE.count() 
nombres = ["Cliente","No Cliente"]
plt.pie(data_final , labels=nombres, autopct="%0.2f %%", colors=colors,
        explode=desfase,radius=5,pctdistance=0.6,rotatelabels=0,startangle=40)
plt.title("FLG_CLIENTE (data original)",fontsize=16, weight="bold")
plt.axis("equal")
plt.show()


# In[64]:


fig= plt.figure(figsize=(30,10))
plt.subplot2grid((2,3),(0,0))
data_inicial=dat.groupby('TARGET_MODEL2').TARGET_MODEL2.count() 
nombres = ["Cliente NO FUGA","Cliente FUGA"]
plt.pie(data_inicial, labels=nombres, autopct="%0.2f %%", colors=colors,
        explode=desfase,radius=5,pctdistance=0.6,rotatelabels=50,startangle=10)
plt.title("TARGET_MODEL2",fontsize=16, weight="bold")
plt.axis("equal")
#------------------------------------------------------------------------
plt.subplot2grid((2,3),(0,1))
target_graf =dat.groupby('TARGET_MODEL2').TARGET_MODEL2.count() 
nombres = ["Cliente NO FUGA","Cliente FUGA"]
plt.pie(target_graf, labels=nombres, autopct="%0.2f %%", colors=colors,
        explode=desfase,radius=5,pctdistance=0.6,rotatelabels=50,startangle=10)
plt.title("TARGET_MODEL2 (data original)",fontsize=16, weight="bold")
plt.axis("equal")
plt.show()


# ### 1.2.2. GRÁFICA DE HISTOGRAMA

# In[65]:


fig= plt.figure(figsize=(30,10))
plt.subplot2grid((2,3),(0,0))
dat['INGRESO_BRUTO_M1'].hist(color=colors[2]).set_title("INGRESO_BRUTO_M1(data limpiada)",
                                                    fontsize=16, weight="bold",color="red")
#--------------------------------------------------------------------------------
plt.subplot2grid((2,3),(0,1))
data_resultante['INGRESO_BRUTO_M1'].hist(color=colors[5]).set_title("INGRESO_BRUTO_M1 (data original)",
                                                           fontsize=16, weight="bold",color="red")
#--------------------------------------------------------------------------------
fig= plt.figure(figsize=(30,10))
plt.subplot2grid((2,3),(1,0))
dat['ANT_CLIENTE'].hist(color=colors[2]).set_title("ANT_CLIENTE (data limpiada)",
                                                    fontsize=16, weight="bold",color="red")
#--------------------------------------------------------------------------------
plt.subplot2grid((2,3),(1,1))
data_resultante['ANT_CLIENTE'].hist(color=colors[5]).set_title("ANT_CLIENTE (data original)",
                                                           fontsize=16, weight="bold",color="red")
plt.show()


# ### 1.2.3. GRAFICAS DE BARRAS

# **VEAMOS LA DISTRIBUSCION DE LOS DEPARTAMENTOS DE LOS CLIENTES**

# In[66]:


departamento_freq=pd.value_counts(dat.DEPARTAMENTO)
departamento_freq


# In[67]:


departamento_freq/sum(departamento_freq)*100


# In[68]:


with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(15,7))
    plot = departamento_freq.plot(kind='barh',rot=0,color=colors)
plot
plt.title('FRECUENCIAS DE LOS DEPARTAMENTOS DE LOS CLIENTES',weight='bold',color='red')
plt.text(498174,0,"63.26%",weight="bold",color="yellow")
plt.text(43021,1,"5.46%",weight="bold",color="yellow")
plt.text(33483,2,"4.25%",weight="bold",color="yellow")
plt.text(28401,3,"3.61%",weight="bold",color="yellow")
plt.text(23374,4,"2.97%",weight="bold",color="yellow")
plt.text(21906,5,"2.78%",weight="bold",color="yellow")
plt.text(21101,6,"2.68%",weight="bold",color="yellow")
plt.text(17426,7,"2.21%",weight="bold",color="yellow")
plt.text(14115,8,"1.79%",weight="bold",color="yellow")
plt.text(13040,9,"1.66%",weight="bold",color="yellow")
plt.text(11041,10,"1.40%",weight="bold",color="yellow")
plt.text(10779,11,"1.37%",weight="bold",color="yellow")
plt.text(10676,12,"1.36%",weight="bold",color="yellow")
plt.text(5548,13,"0.70%",weight="bold",color="yellow")
plt.text(5410,14,"0.69%",weight="bold",color="yellow")
plt.text(4888,15,"0.62%",weight="bold",color="yellow")
plt.text(4790,16,"0.61%",weight="bold",color="yellow")
plt.text(4259,17,"0.54%",weight="bold",color="yellow")
plt.text(4038,18,"0.51%",weight="bold",color="yellow")
plt.text(3411,19,"0.43%",weight="bold",color="yellow")
plt.text(2809,20,"0.36%",weight="bold",color="yellow")
plt.text(2452,21,"0.31%",weight="bold",color="yellow")
plt.text(2108,22,"0.27%",weight="bold",color="yellow")
plt.text(705,23,"0.09%",weight="bold",color="yellow")
plt.text(540,24,"0.07%",weight="bold",color="yellow")
plt.show()


# **VEAMOS LA DISTRIBUCIÓN DE LA EDAD DE LOS CLIENTES PERO AGRUPADOS**

# VAMOS AGRUPAR DE LA SIGUIENTE MANERA:

#  1. MENORES DE EDAD: (0, 18]
#  2. MAYORES DE EDAD: (18, 65]
#  3. ANCIANOS: (65, 114]

# **SUPERVIVENCIA DESAGREGADA POR GÉNERO**

# In[69]:


with plt.style.context('dark_background'):
    fig= plt.figure(figsize=(30,10))
    plt.subplot2grid((2,3),(0,0))
    sns.countplot(y='SEXO', hue='TARGET_MODEL2', data=dat,palette=colors)
plt.title('Supervivencia desagregada por genero',weight='bold',color='red')
plt.show()


# **VEAMOS LA DISTRIBUCIÓN DE LA EDAD DE LOS CLIENTES PERO AGRUPADOS**

# VAMOS AGRUPAR DE LA SIGUIENTE MANERA:

#  1. MENORES DE EDAD: (0, 18]
#  2. MAYORES DE EDAD: (18, 65]
#  3. ANCIANOS: (65, 114]

# In[70]:


dat["EDAD_CAT"]=pd.cut(dat.EDAD,bins=[dat.EDAD.min(),18,65,dat.EDAD.max()])
dat.head()


# In[71]:


edad=dat.groupby("EDAD_CAT").size()
edad


# In[72]:


e=round((edad/sum(edad))*100,2)
print("El % de datos es: " )
print("-------------", e)


# In[73]:


with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(8,7))
    plot =edad.plot(kind='bar', 
                         rot=0,color=colors) 
plot
plt.title('FRECUENCIAS DE LA EDAD CATEGORIZADA',weight='bold',color='red')
plt.text(-0.2,798,"0.10%",weight="bold",color="yellow")
plt.text(0.8,752244,"95.53%",weight="bold",color="yellow")
plt.text(1.8,34412,"4.37%",weight="bold",color="yellow")
plt.show()


# **VEAMOS LA DISTRIBUCIÓN DEL INGRESO BRUTO DE LOS CLIENTES PERO AGRUPADOS**

# LO AGRUPAMOS DE LA SIGUIENTE MANERA:

# 1. CLASE POBRE: (681.0, 1200.0]
# 2. CLASE MEDIA BAJA: (1200.0, 5000.0]
# 3. CLASE MEDIA ALTA: (5000.0, 10000.0]
# 4. CLASE ALTA: (10000.0, 214284.0]

# In[74]:


dat_1=pd.read_csv("TRAIN_FUGA_COMPLETO.csv",sep=",", encoding="ISO-8859-1")
dat_1['INGRESO_BRUTO_M1_CAT']=pd.cut(dat_1.INGRESO_BRUTO_M1,bins=[dat_1.INGRESO_BRUTO_M1.min(),1200,5000,10000,dat_1.INGRESO_BRUTO_M1.max()])
dat_1.head()


# In[75]:


ingreso=dat_1.groupby('INGRESO_BRUTO_M1_CAT').size()
ingreso


# In[76]:


p=round((ingreso/sum(ingreso))*100,2)
print("El % de datos es: " )
print("--------------------", p)


# In[77]:


with plt.style.context('dark_background'):
    fig = plt.figure(figsize=(8,7))
    plot =ingreso.plot(kind='bar', 
                         rot=0,color=colors) 
plot
plt.title('FRECUENCIAS DEL INGRESO BRUTO CATEGORIZADO',weight='bold',color='red')
plt.text(-0.2,206224,"26.19%",weight="bold",color="yellow")
plt.text(0.8,514384,"65.32%",weight="bold",color="yellow")
plt.text(1.8,49208,"6.25%",weight="bold",color="yellow")
plt.text(2.8,17671,"2.24%",weight="bold",color="yellow")
plt.show()


# **VEAMOS LA DESINDAD DEL INGRESO BRUTO**

# In[78]:


warnings.filterwarnings("ignore")
dat_2=pd.read_csv("TRAIN_FUGA_COMPLETO.csv",sep=",", encoding="ISO-8859-1")
with plt.style.context('dark_background'):
    ax1=sns.distplot(dat_2['INGRESO_BRUTO_M1'])
plt.show()


# DEBIDO A LA CANTIDAD DE LOS DATOS DE NUESTRO DATAFRAME NO SE PUEDE APRECIAR BIEN LA SIMETRÍA O ASIMETRÍA DE LA VARIABLE INGRESO BRUTO , PARA ELLO VAMOS A TOMAR UNA MUESTRA DE LOS DATOS OBTENIDOS PARA PODER REALIZAR UN MEJOR ANÁLISIS.

# In[79]:


dat_2=pd.read_csv("TRAIN_FUGA_COMPLETO.csv",sep=",", encoding="ISO-8859-1")
medio=(dat_2['INGRESO_BRUTO_M1']>2500)&(dat_2['INGRESO_BRUTO_M1']<18000)
medio


# In[80]:


nuevo=dat_2[medio]
nuevo.head(10)


# In[81]:


nuevo=dat_2[medio]
nuevo.shape


# In[82]:


mediana=nuevo["INGRESO_BRUTO_M1"].median()
media=nuevo["INGRESO_BRUTO_M1"].mean()
moda=nuevo["INGRESO_BRUTO_M1"].mode()
print("La mediana:",mediana)
print("La media:",media)
print("La moda:",moda)
minimo=nuevo["INGRESO_BRUTO_M1"].min()
print("El valor mínimo:",minimo)
maximo=nuevo["INGRESO_BRUTO_M1"].max()
print("El valor máximo",maximo)


# In[83]:


warnings.filterwarnings("ignore")
with plt.style.context('dark_background'):
    ax2=sns.distplot(nuevo['INGRESO_BRUTO_M1'])
plt.title("DENSIDAD DEL INGRESO",weight='bold',color='red')
plt.show()


# PODEMOS VER QUE TIENE UNA ASIMETRíA POSITIVA DE LA VARIABLE INGRESO BRUTO

# ### 1.2.4. GRÁFICA DE DISPERCIÓN

# **VEAMOS SI EXISTE UNA CORRELACIÓN ENTRE LAS VARIABLES**

# In[84]:


with plt.style.context('dark_background'):
    nuevo.plot.scatter(x="DEPARTAMENTO",
                 y="INGRESO_BRUTO_M1",
                 alpha=0.7,rot=90) #sombreado de los puntos, menor valor es más claro.
plt.title("GRÁFICA DE DISPERCIÓN DE LOS DEPARTAMENTOS VS INGRESO BRUTO",weight='bold',color='red')
plt.show()


# In[85]:


columnas=nuevo.columns
sns.pairplot(nuevo[columnas], #data y sus columnas seleccionadas
             height = 4.5) #tamaño de la gráfica
plt.show()


# PODEMOS OBSERVAR QUE NO EXISTE UNA CORRELACION ENTRE LOS DATOS.

# ### GRÁFICOS INFORMATIVOS

# In[86]:


fig= plt.figure(figsize=(30,10))
#with plt.style.context('dark_background'): 
plt.subplot2grid((2,3),(0,0))
sns.countplot(x='INGRESO_BRUTO_M1_CAT', hue='TARGET_MODEL2', data=dat_1)
plt.title('Supervivencia desagregada por ingreso',fontsize=16, weight="bold",color='red')
plt.legend(loc="upper right")

plt.subplot2grid((2,3),(0,1))
ax=sns.distplot(nuevo['INGRESO_BRUTO_M1'])
plt.title('Densidad ingreso',fontsize=16, weight="bold",color='red')

plt.subplot2grid((2,3),(1,0))
dat_1.INGRESO_BRUTO_M1_CAT[dat_2.TARGET_MODEL2 == 1].value_counts().plot(kind="barh", color = colors)
plt.title("Supervivencia Ingreso respecto al Target",fontsize=16, weight="bold",color='red')

plt.subplot2grid((2,3),(1,1))
dat_1.INGRESO_BRUTO_M1_CAT[dat_2.ANT_CLIENTE == 1].value_counts(normalize=True).plot(kind="barh",color = colors)
plt.title("Supervivenvcia Ingreso respecto a la Antiguedad del Cliente",fontsize=16, weight="bold",color='red')

plt.show()


# In[87]:


fig= plt.figure(figsize=(30,10))
plt.subplot2grid((2,3),(0,0))
sns.countplot(x='EDAD_CAT', hue='TARGET_MODEL2', data=dat)
plt.title('Supervivencia desagregada por genero',fontsize=16, weight="bold",color='red')

plt.subplot2grid((2,3),(0,1))
ax=sns.distplot(dat_2['EDAD'])
plt.title('Densidad Edad',fontsize=16, weight="bold",color='red')

plt.subplot2grid((2,3),(1,0))
dat.EDAD_CAT[dat_2.TARGET_MODEL2 == 1].value_counts().plot(kind="barh", color = colors)
plt.title("Supervivencia Edad respecto al Target",fontsize=16, weight="bold",color='red')

plt.subplot2grid((2,3),(1,1))
dat.EDAD_CAT[dat_2.ANT_CLIENTE == 1].value_counts(normalize=True).plot(kind="barh",color = colors)
plt.title("Supervivenvcia Edad respecto a la Antiguedad del Cliente",fontsize=16, weight="bold",color='red')

plt.show()


# In[ ]:





# ## 2. Realizar un análisis exploratorio sobre presencia de outliers. 

# In[88]:


from IPython.display import Image
get_ipython().run_line_magic('matplotlib', 'inline')
Image(filename='E:\PYTHOM\MODULO 1\EXAMEN FINAL-MODULO1/interpretacion_boxplot.png', width=600)


# #### Graficando antes de eliminar los Outliers

# **BOXPLOT DEL SEXO SEGÚN EL INGRESO BRUTO**

# In[89]:


with plt.style.context('dark_background'):
    sns.boxplot(x=nuevo['SEXO'], #Será la variable categorizadora o separadora.
            y=nuevo['INGRESO_BRUTO_M1']) #La variable cuantitativa de rpta.
plt.title("Boxplot del SEXO según INGRESO BRUTO",weight='bold',color='red')
plt.show()


# In[90]:


mediana=nuevo[nuevo.SEXO == 'M'].INGRESO_BRUTO_M1.median()
Q1=nuevo[nuevo.SEXO == 'M'].INGRESO_BRUTO_M1.quantile(0.25)
Q3=nuevo[nuevo.SEXO == 'M'].INGRESO_BRUTO_M1.quantile(0.75)
BS=Q3+1.5*(Q3-Q1)
BI=Q3-1.5*(Q3-Q1)
print("INFORMACIÓN DEL GÉNERO MASCULINO SEGÚN EL INGRESO BRUTO\n")
print("Primer cuartil:",Q1)
print("Tercer cuartil:",Q3)
print("La mediana:",mediana)
print("Bigote superior de mi boxplot es: ", BS)
print("Bigote inferior de mi boxplot es: ", BI)
print("------------------------------------------------------------")
mediana=nuevo[nuevo.SEXO == 'F'].INGRESO_BRUTO_M1.median()
Q1=nuevo[nuevo.SEXO == 'F'].INGRESO_BRUTO_M1.quantile(0.25)
Q3=nuevo[nuevo.SEXO == 'F'].INGRESO_BRUTO_M1.quantile(0.75)
BS=Q3+1.5*(Q3-Q1)
BI=Q3-1.5*(Q3-Q1)
print("INFORMACIÓN DEL GÉNERO FEMENINO SEGÚN EL INGRESO BRUTO\n")
print("Primer cuartil:",Q1)
print("Tercer cuartil:",Q3)
print("La mediana:",mediana)
print("Bigote superior de mi boxplot es: ", BS)
print("Bigote inferior de mi boxplot es: ", BI)


# **BOXPLOT DEL INGRESO BRUTO**

# In[91]:


with plt.style.context('dark_background'):
    sns.boxplot(nuevo.INGRESO_BRUTO_M1)
plt.title("GRÁFICA DE CAJA DEL INGRESO BRUTO",weight='bold',color='red')
plt.show()


# In[92]:


print("INFORMACIÓN DEL BOXPLOT DEL INGRESO BRUTO\n")
Q1=nuevo.INGRESO_BRUTO_M1.quantile(0.25)
print("Primer cuartil:",Q1)
Q3=nuevo.INGRESO_BRUTO_M1.quantile(0.75)
print("Tercer cuartil:",Q3)
Rango_inter_cuart=Q3-Q1
print("El rango intercuartil:",Rango_inter_cuart)
mediana=nuevo.INGRESO_BRUTO_M1.median()
print("La mediana:",mediana)
minimo=nuevo.INGRESO_BRUTO_M1.min()
print("El valor mínimo:",minimo)
maximo=nuevo.INGRESO_BRUTO_M1.max()
print("El valor máximo",maximo)
N=Q3+1.5*(Q3-Q1)
BS=Q3+1.5*(Q3-Q1)
BI=Q3-1.5*(Q3-Q1)
print("Bigote superior de mi boxplot es: ", BS)
print("Bigote inferior de mi boxplot es: ", BI)


# **BOXPLOT DE LA EDAD**

# In[93]:


with plt.style.context('dark_background'):
    sns.boxplot(dat_2.EDAD)
plt.title("GRÁFICA DE CAJA DE LA EDAD",weight='bold',color='red')
plt.show()


# In[94]:


print("INFORMACIÓN DEL BOXPLOT DE LA EDAD\n")
Q1=nuevo.EDAD.quantile(0.25)
print("Primer cuartil:",Q1)
Q3=nuevo.EDAD.quantile(0.75)
print("Tercer cuartil:",Q3)
Rango_inter_cuart=Q3-Q1
print("El rango intercuartil:",Rango_inter_cuart)
mediana=nuevo.EDAD.median()
print("La mediana:",mediana)
minimo=nuevo.EDAD.min()
print("El valor mínimo:",minimo)
maximo=nuevo.EDAD.max()
print("El valor máximo",maximo)
N=Q3+1.5*(Q3-Q1)
BS=Q3+1.5*(Q3-Q1)
BI=Q3-1.5*(Q3-Q1)
print("Bigote superior de mi boxplot es: ", BS)
print("Bigote inferior de mi boxplot es: ", BI)


# #### Eliminando los Outliers

# In[95]:


dat_3=pd.read_csv("TRAIN_FUGA_COMPLETO.csv",sep=",", encoding="ISO-8859-1")
dat_3.head(50)


# ##### IQR score

# In[96]:


Q1 = dat_3.quantile(0.25)
Q3 = dat_3.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[97]:


print((dat_3 < (Q1 - 1.5 * IQR)) | (dat_3 > (Q3 + 1.5 * IQR)))


# In[98]:


dat_3_out= dat_3[~((dat_3 < (Q1 - 1.5 * IQR)) |(dat_3 > (Q3 + 1.5 * IQR))).any(axis=1)]
dat_3_out.head(50)


# In[99]:


len(dat_3_out)


# In[100]:


len(dat_3)


# #### Graficando despues de eliminar los Outliers

# **BOXPLOT DEL SEXO SEGÚN EL INGRESO BRUTO**

# In[101]:


with plt.style.context('dark_background'):
    sns.boxplot(x=dat_3_out['SEXO'], #Será la variable categorizadora o separadora.
            y=dat_3_out['INGRESO_BRUTO_M1']) #La variable cuantitativa de rpta.
plt.title("Boxplot del SEXO según INGRESO BRUTO",weight='bold',color='red')
plt.show()


# In[102]:


print("INFORMACIÓN DEL GÉNERO MASCULINO SEGÚN EL INGRESO BRUTO\n")
mediana=nuevo[nuevo.SEXO == 'M'].INGRESO_BRUTO_M1.median()
Q1=nuevo[nuevo.SEXO == 'M'].INGRESO_BRUTO_M1.quantile(0.25)
Q3=nuevo[nuevo.SEXO == 'M'].INGRESO_BRUTO_M1.quantile(0.75)
BS=Q3+1.5*(Q3-Q1)
BI=Q3-1.5*(Q3-Q1)
print("Primer cuartil:",Q1)
print("Tercer cuartil:",Q3)
print("La mediana:",mediana)
print("Bigote superior de mi boxplot es: ", BS)
print("Bigote inferior de mi boxplot es: ", BI)

print("INFORMACIÓN DEL GÉNERO FEMENINO SEGÚN EL INGRESO BRUTO\n")
mediana=nuevo[nuevo.SEXO == 'F'].INGRESO_BRUTO_M1.median()
Q1=nuevo[nuevo.SEXO == 'F'].INGRESO_BRUTO_M1.quantile(0.25)
Q3=nuevo[nuevo.SEXO == 'F'].INGRESO_BRUTO_M1.quantile(0.75)
BS=Q3+1.5*(Q3-Q1)
BI=Q3-1.5*(Q3-Q1)
print("Primer cuartil:",Q1)
print("Tercer cuartil:",Q3)
print("La mediana:",mediana)
print("Bigote superior de mi boxplot es: ", BS)
print("Bigote inferior de mi boxplot es: ", BI)


# **BOXPLOT DE LA EDAD**

# In[103]:


with plt.style.context('dark_background'):
    sns.boxplot(dat_3_out.EDAD)
plt.title("GRÁFICA DE CAJA DE LA EDAD",weight='bold',color='red')
plt.show()


# In[104]:


print("INFORMACIÓN DEL BOXPLOT DE LA EDAD\n")
Q1=dat_3_out.EDAD.quantile(0.25)
print("Primer cuartil:",Q1)
Q3=dat_3_out.EDAD.quantile(0.75)
print("Tercer cuartil:",Q3)
Rango_inter_cuart=Q3-Q1
print("El rango intercuartil:",Rango_inter_cuart)
mediana=dat_3_out.EDAD.median()
print("La mediana:",mediana)
minimo=dat_3_out.EDAD.min()
print("El valor mínimo:",minimo)
maximo=dat_3_out.EDAD.max()
print("El valor máximo",maximo)
N=Q3+1.5*(Q3-Q1)
BS=Q3+1.5*(Q3-Q1)
BI=Q3-1.5*(Q3-Q1)
print("Bigote superior de mi boxplot es: ", BS)
print("Bigote inferior de mi boxplot es: ", BI)


# **BOXPLOT DEL INGRESO BRUTO**

# In[105]:


with plt.style.context('dark_background'):
    sns.boxplot(dat_3_out.INGRESO_BRUTO_M1)
plt.title("GRÁFICA DE CAJA DEL INGRESO BRUTO",weight='bold',color='red')
plt.show()


# In[106]:


print("INFORMACIÓN DEL BOXPLOT DEL INGRESO BRUTO\n")
Q1=dat_3_out.INGRESO_BRUTO_M1.quantile(0.25)
print("Primer cuartil:",Q1)
Q3=dat_3_out.INGRESO_BRUTO_M1.quantile(0.75)
print("Tercer cuartil:",Q3)
Rango_inter_cuart=Q3-Q1
print("El rango intercuartil:",Rango_inter_cuart)
mediana=dat_3_out.INGRESO_BRUTO_M1.median()
print("La mediana:",mediana)
minimo=dat_3_out.INGRESO_BRUTO_M1.min()
print("El valor mínimo:",minimo)
maximo=dat_3_out.INGRESO_BRUTO_M1.max()
print("El valor máximo",maximo)
N=Q3+1.5*(Q3-Q1)
BS=Q3+1.5*(Q3-Q1)
BI=Q3-1.5*(Q3-Q1)
print("Bigote superior de mi boxplot es: ", BS)
print("Bigote inferior de mi boxplot es: ", BI)


# Podemos vizualizar que acurrido algunos ajustes con las gráficas de cajas y con la información de las cajas y esto ocurrio ya que se eliminio los valores atípicos de nuestro DataFrame

# ## 3. Realizar una discretización de las variables : INGRESO_BRUTO_M1 y EDAD teniendo en cuenta al menos dos técnicas de discretización no supervisada y agregar las variables discretizadas a nuestro conjunto de datos original

# In[107]:


from sklearn.preprocessing import KBinsDiscretizer


# In[108]:


dat_2.info()


# ### 3.1 Descretización por intervalos de igual amplitud

# **PARA LA VARIABLA EDAD**

# In[109]:


n=len(dat_2)
k=1+math.log2(n)
k=round(k,0)
k


# In[110]:


amplitud=KBinsDiscretizer(n_bins=21,encode="ordinal",strategy="uniform")
nueva_dat_2=amplitud.fit_transform(dat_2[['EDAD']])
dat_2["EDAD_amplitud"]=nueva_dat_2
dat_2["EDAD_amplitud"]=dat_2["EDAD_amplitud"].astype(np.int64)
dat_2.info()


# In[111]:


graf_edad_amplitud=dat_2.groupby(dat_2.EDAD_amplitud).size()
graf_edad_amplitud


# In[112]:


graf_edad_amplitud/sum(graf_edad_amplitud)


# In[113]:


with plt.style.context('dark_background'):
    graf_edad_amplitud.plot(kind="bar", rot=0,color=colors)
plt.title("DISTRIBUCIÓN DE LA EDAD POR IGUALDAD DE AMPLITUD",weight='bold',color='red')
plt.show()


# **PARA LA VARIABLE INGRESO BRUTO**

# In[114]:


amplitud_ingreso=KBinsDiscretizer(n_bins=21,encode="ordinal",strategy="uniform")
nueva_dat_2_ingreso=amplitud_ingreso.fit_transform(dat_2[['INGRESO_BRUTO_M1']])
dat_2["INGRESO_BRUTO_M1_amplitud"]=nueva_dat_2_ingreso
dat_2["INGRESO_BRUTO_M1_amplitud"]=dat_2["INGRESO_BRUTO_M1_amplitud"].astype(np.int64)
dat_2.info()


# In[115]:


graf_ingreso_amplitud=dat_2.groupby(dat_2.INGRESO_BRUTO_M1_amplitud).size()
graf_ingreso_amplitud


# In[116]:


graf_ingreso_amplitud/sum(graf_ingreso_amplitud)


# In[117]:


with plt.style.context('dark_background'):
    graf_ingreso_amplitud.plot(kind="bar", rot=0,color=colors)
plt.title("DISTRIBUCIÓN DEL INGRESO BRUTO POR IGUALDAD DE AMPLITUD",weight='bold',color='red')
plt.show()


# ### 3.2 Discretización por Cuantiles

# **PARA LA VARIABLA EDAD**

# In[118]:


quartil=KBinsDiscretizer(n_bins=4, encode="ordinal",strategy="quantile")
nuevo_dat_2_cuantil=quartil.fit_transform(dat_2[['EDAD']])
dat_2["EDAD_quartil"]=nuevo_dat_2_cuantil
dat_2["EDAD_quartil"]=dat_2["EDAD_quartil"].astype(np.int64)
dat_2.info()


# In[119]:


graf_edad_quartil=dat_2.groupby(dat_2.EDAD_quartil).size()
graf_edad_quartil


# In[120]:


graf_edad_quartil/sum(graf_edad_quartil)*100


# In[121]:


with plt.style.context('dark_background'):
    graf_edad_quartil.plot(kind="bar", rot=0,color=colors)
plt.title("DISTRIBUCIÓN DE LA EDAD POR IGUALDAD DE QUARTIL",weight='bold',color='red')
plt.text(-0.3,171568,"21.79%",weight="bold",color="yellow")
plt.text(0.7,211373,"26.84%",weight="bold",color="yellow")
plt.text(1.7,200973,"25.52%",weight="bold",color="yellow")
plt.text(2.7,203581,"25.85%",weight="bold",color="yellow")
plt.show()


# **PARA LA VARIABLE INGRESO BRUTO**

# In[122]:


quartil_ingreso=KBinsDiscretizer(n_bins=4,encode="ordinal",strategy="quantile")
nueva_dat_2_quartil_ingreso=quartil_ingreso.fit_transform(dat_2[['INGRESO_BRUTO_M1']])
dat_2["INGRESO_BRUTO_M1_quartil"]=nueva_dat_2_quartil_ingreso
dat_2["INGRESO_BRUTO_M1_quartil"]=dat_2["INGRESO_BRUTO_M1_quartil"].astype(np.int64)
dat_2.info()


# In[123]:


graf_ingreso_quartil=dat_2.groupby(dat_2.INGRESO_BRUTO_M1_quartil).size()
graf_ingreso_quartil


# In[124]:


graf_ingreso_quartil/sum(graf_ingreso_quartil)*100


# In[125]:


graf_ingreso_quartil=dat_2.groupby(dat_2.INGRESO_BRUTO_M1_quartil).size()
with plt.style.context('dark_background'):
    graf_ingreso_quartil.plot(kind="bar", rot=0,color=colors)
plt.title("DISTRIBUCIÓN DEL INGRESO BRUTO POR IGUALDAD DE QUARTIL",weight='bold',color='red')
plt.text(-0.3,196597,"24.96%",weight="bold",color="yellow")
plt.text(0.7,197066," 25.02%",weight="bold",color="yellow")
plt.text(1.7,196855,"25.00%",weight="bold",color="yellow")
plt.text(2.7,196977," 25.01%",weight="bold",color="yellow")
plt.show()


# ### 3.3. Discretización por KMeans

# **PARA LA VARIABLA EDAD**

# In[126]:


kmeans=KBinsDiscretizer(n_bins=6,encode="ordinal",strategy="kmeans")
nuevo_dat_2_kmeans=kmeans.fit_transform(dat_2[['EDAD']])
dat_2["EDAD_kmeans"]=nuevo_dat_2_kmeans
dat_2["EDAD_quartil"]=dat_2["EDAD_kmeans"].astype(np.int64)
dat_2.info()


# **Hallando la frecuencia de la EDAD y su porcentaje**

# In[127]:


graf_edad_kmeans=dat_2.groupby(dat_2.EDAD_kmeans).size()
graf_edad_kmeans


# In[128]:


graf_edad_kmeans/sum(graf_edad_kmeans)*100


# In[129]:


with plt.style.context('dark_background'):
    graf_edad_kmeans.plot(kind="bar", rot=0,color=colors)
plt.title("DISTRIBUCIÓN DE LA EDAD POR KMEANS",weight='bold',color='red')
plt.text(-0.3, 200067,"25.41%",weight="bold",color="yellow")
plt.text(0.7, 254058,"32.26%",weight="bold",color="yellow")
plt.text(1.7,182762,"23.21%",weight="bold",color="yellow")
plt.text(2.7,98509,"12.51%",weight="bold",color="yellow")
plt.text(3.7,37502,"4.76%",weight="bold",color="yellow")
plt.text(4.7,14597,"1.85%",weight="bold",color="yellow")
plt.show()


# **PARA LA VARIABLE INGRESO BRUTO**

# In[130]:


kmeans_ingreso=KBinsDiscretizer(n_bins=6,encode="ordinal",strategy="kmeans")
nuevo_dat_2_kmeans_ingreso=kmeans_ingreso.fit_transform(dat_2[['INGRESO_BRUTO_M1']])
dat_2["INGRESO_BRUTO_M1_kmeans"]=nuevo_dat_2_kmeans_ingreso
dat_2["INGRESO_BRUTO_M1_quartil"]=dat_2["INGRESO_BRUTO_M1_kmeans"].astype(np.int64)
dat_2.info()


# **Hallando la frecuencia del INGRESO BRUTO y su porcentaje**

# In[131]:


graf_ingreso_kmeans=dat_2.groupby(dat_2.INGRESO_BRUTO_M1_kmeans).size()
graf_ingreso_kmeans


# In[132]:


graf_ingreso_kmeans/sum(graf_ingreso_kmeans)*100


# In[133]:


with plt.style.context('dark_background'):
    graf_ingreso_kmeans.plot(kind="bar", rot=0,color=colors)
plt.title("DISTRIBUCIÓN DEL INGRESO BRUTO POR KMEANS",weight='bold',color='red')
plt.text(-0.3, 626331,"79.53%",weight="bold",color="yellow")
plt.text(0.7,112544,"14.29%",weight="bold",color="yellow")
plt.text(1.7,37818,"4.80%",weight="bold",color="yellow")
plt.text(2.7, 9092,"1.15%",weight="bold",color="yellow")
plt.text(3.7,1555,"0.20%",weight="bold",color="yellow")
plt.text(4.7,155 ,"0.02%",weight="bold",color="yellow")
plt.show()


# **Mostramos los resultados de la data dicretizadas en nuestra data original**

# In[134]:


dat_2.head()


# **Guardanos**

# In[135]:


dat_2.to_csv("TRAIN_FUGA_COMPLETO_DISCRETIZADA.csv", index=False)
Image(filename='E:\PYTHOM\MODULO 1\EXAMEN FINAL-MODULO1/Captura de pantalla 2021-03-09 183235.png', width=900)


# ## 4. Aplicar dos técnicas de balanceo de datos a nuestra variable TARGET (objetivo) y agregarlas a nuestro conjunto de datos original. Use los parámetros vistos en clase. (3 puntos)

# In[136]:


import scipy
import sklearn
import imblearn


# In[137]:


from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import RandomOverSampler
from imblearn.combine import SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.model_selection import train_test_split
import random


# In[138]:


archivo_csv="TRAIN_FUGA_COMPLETO.csv"
datos=pd.read_csv(archivo_csv,sep=",", encoding="ISO-8859-1")
datos.head(50)


# **Sacando las columnas analizar**

# In[139]:


dat_analisis = datos[['EDAD','SEXO','INGRESO_BRUTO_M1','FREC_BPI_TD','FLG_ADEL_SUELDO_M1',
            'PROM_CTD_TRX_6M','FREC_AGENTE','FREC_KIOSKO','ANT_CLIENTE','CTD_RECLAMOS_M1']]
dat_analisis.head()


# In[140]:


dat_analisis.shape


# **Vizualicemos los tipos de variable**

# In[141]:


dat_analisis.dtypes


# **Transformemos la variable SEXO que es de tipo object a tipo float**

# In[142]:


dat_analisis["SEXO"]=dat_analisis["SEXO"].replace("F","0")
dat_analisis["SEXO"]=dat_analisis["SEXO"].replace("M","1")
dat_analisis["SEXO"]=dat_analisis["SEXO"].astype(float)


# **Mostramos los resultados**

# In[143]:


dat_analisis.dtypes


# **Agrupando columnas por tipo de datos**

# In[144]:


dat_analisis.columns.to_series().groupby(dat_analisis.dtypes).groups


# In[145]:


tipos = dat_analisis.columns.to_series().groupby(dat_analisis.dtypes).groups


# **Armando lista de columnas categóricas**

# In[146]:


entero = tipos[np.dtype("int64")]
print("Los números de tipo entero son:\n",entero )

flotante = tipos[np.dtype("float")]
print("Los números de tipo flotante son:\n",flotante )


# **Transformamos los enteros a flotantes**

# In[147]:


for c in entero:
    dat_analisis[c]=dat_analisis[c].astype(float)  


# In[148]:


dat_analisis.dtypes


# **Concatenamos dat_analisis y datos['TARGET_MODEL2']**

# In[149]:


dat_analisis_2 = pd.concat([dat_analisis, datos['TARGET_MODEL2']], axis=1)
dat_analisis_2.head()


# **Visualizamos los tipo**

# In[150]:


dat_analisis_2.dtypes


# **Hallemos las frecuencias de la variable TARGET_MODEL2**

# In[151]:


frec_datos=pd.value_counts(dat_analisis_2.TARGET_MODEL2,sort=True)
frec_datos


# In[152]:


frec_datos/sum(frec_datos)*100


# In[153]:


with plt.style.context('dark_background'):
    pd.value_counts(dat_analisis_2.TARGET_MODEL2).plot(kind='bar',rot=0,color=colors)
plt.title("FRECUENCIA DE NÚMEROS DE OBSERVACIONES",weight='bold',color='red')
plt.xlabel("Ocurrencia de Incidentes")
plt.ylabel("TARGET_MODEL2")
plt.text(-0.2,743749,"94.444917%",weight="bold",color="yellow")
plt.text(0.8,43746,"5.555083%",weight="bold",color="yellow")
plt.show()


# In[154]:


dat_analisis_2.info()


# **Particionamiento de datos**

# In[155]:


X, y = dat_analisis_2.iloc[:, 0:10].values, dat_analisis_2.iloc[:,10].values

X_train, X_test, y_train, y_test =    train_test_split(X, #valores de los predictores
                     y, #los valores del target
                     test_size=0.4, #proporción para datos de testeo
                     random_state=2021, #semilla
                     stratify=y) #la variable de estratificación


# **Datos de entrenamiento**

# In[156]:


x_t= pd.DataFrame(X_train, columns=['EDAD','SEXO','INGRESO_BRUTO_M1','FREC_BPI_TD','FLG_ADEL_SUELDO_M1',
            'PROM_CTD_TRX_6M','FREC_AGENTE','FREC_KIOSKO','ANT_CLIENTE','CTD_RECLAMOS_M1'])
y_t= pd.DataFrame(y_train, columns=['TARGET_MODEL2'])

dat_analisis_2_entrenados = pd.concat([x_t, y_t], axis=1)
dat_analisis_2_entrenados.head()


# In[157]:


dat_analisis_2_entrenados.shape


# **Hallemos las frecuencias de la variable TARGET_MODEL2 ya entrenada**

# In[158]:


count_classes=pd.value_counts(dat_analisis_2_entrenados.TARGET_MODEL2,sort=True)
count_classes


# In[159]:


count_classes/sum(count_classes)*100


# In[160]:


with plt.style.context('dark_background'):
    count_classes.plot(kind = 'bar',rot=0,color=colors)
plt.title("FRECUENCIA DE NÚMEROS DE OBSERVACIONES",weight='bold',color='red')
plt.xlabel("TARGET_MODEL2")
plt.ylabel("número de observaciones")
plt.text(-0.2,446249," 94.444917%",weight="bold",color="yellow")
plt.text(0.8,26248,"5.555168%",weight="bold",color="yellow")
plt.show()


# In[161]:


dat_analisis_2_entrenados.info()


# ### 4.1. OVERSAMPLING

# In[162]:


pip install tensorflow


# In[163]:


get_ipython().system('pip install --user imbalanced-learn')


# In[164]:


pip install --user git+https://github.com/scikit-learn-contrib/imbalanced-learn.git


# In[165]:


over=RandomOverSampler(sampling_strategy=0.7,random_state=2021)


# In[166]:


Xtrain_over, ytrain_over =over.fit_resample(x_t, y_t)


# In[167]:


dat_analisis_2_entrenados_over=pd.concat([Xtrain_over,ytrain_over],axis=1)
dat_analisis_2_entrenados_over


# In[168]:


count_classes_over = pd.value_counts(dat_analisis_2_entrenados_over['TARGET_MODEL2'], sort = True)
count_classes_over


# In[169]:


count_classes_over/sum(count_classes_over)*100


# In[170]:


prop=round(count_classes_over[1]*100/count_classes_over[0],1)
prop


# In[171]:


with plt.style.context('dark_background'):
    count_classes_over.plot(kind = 'bar', 
                   rot=0,color=colors)
plt.xticks(range(2))
plt.title("GRÁFICO OVERSAMPLING",weight="bold",color="red")
plt.xlabel("ocurrencia de incidentes")
plt.ylabel("número de observaciones")
plt.text(-0.1,446249,"58.82%",weight="bold",color="yellow")
plt.text(0.9,312374,"41.18%",weight="bold",color="yellow")
plt.show()


# ### 4.2. SMOTETomek

# In[172]:


Smot=SMOTETomek(sampling_strategy=0.7,random_state=2021)


# In[173]:


Xtrain_Smot, ytrain_Smot =Smot.fit_resample(x_t, y_t)


# In[174]:


dat_analisis_2_entrenamiento_Smot=pd.concat([Xtrain_Smot,ytrain_Smot],axis=1)
dat_analisis_2_entrenamiento_Smot


# In[175]:


count_classes_Smot = pd.value_counts(dat_analisis_2_entrenamiento_Smot['TARGET_MODEL2'], sort = True)
count_classes_Smot


# In[176]:


count_classes_Smot/sum(count_classes_Smot)*100


# In[177]:


prop=round(count_classes_Smot[1]*100/count_classes_Smot[0],1)
prop


# In[178]:


with plt.style.context('dark_background'):
    count_classes_Smot.plot(kind = 'bar',#bar: gráfico de barras
                   rot=0,color =colors)#0 = no rotación de las etiquetas del eje x
plt.xticks(range(2))
plt.title("GRÁFICO SMOTE TOMEK",weight="bold",color="red")
plt.xlabel("ocurrencia de incidentes")
plt.ylabel("número de observaciones")
plt.text(-0.1,440532,"58.96%",weight="bold",color="yellow")
plt.text(0.9,306657,"41.04%",weight="bold",color="yellow")
plt.show()


# # CASO 2:
# La presente aplicación captura datos socioeconómicos a nivel distrital para la realización de un ejemplo de reducción de dimensiones haciendo uso del análisis de componentes principales y factorial.
# Las variables a reducir son: porcentaje de hogares sin medios de comunicación (porc_hogares_sin_medios), porcentaje de alfabetismo (alfabetismo), porcentaje de hogares con 2 o más necesidades básicas incubiertas (porc_2_nbi), índice de desarrollo humano (IDH) y el coeficiente de desigualdad de GINI (GINI).
# 1.	Realizar un análisis de componentes principales para reducción de la dimensionalidad (4 puntos)
# 2.	Realizar un análisis factorial para reducción de la dimensionalidad (4 puntos)
# 

# Las librerias para Los Componentes Principales son:

# In[179]:


import scipy.stats as stats#Para calculo de probabilidades estadisticos
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA#Para descomposición de varianza en el PCA
from sklearn.preprocessing import MinMaxScaler#Para la normalización de dato


# Importamos nuestro base de datos

# In[180]:


os.chdir("E:\PYTHOM\MODULO 1\EXAMEN FINAL-MODULO1")#direccionando la ruta
archivo_spss="AusentismoPres2011.sav"
df=pd.read_spss(path=archivo_spss)
df.head()


# In[181]:


df.info()


# hacemos una pequeña limpieza de datos

# In[182]:


data=df.dropna(subset=['GINI'])
data.info()


# ### PARTICIONANDO LOS DATOS

# In[183]:


x=data.iloc[:, [10,11,12,14,15]].values
y=data.iloc[:, 9].values

#Dividimos un conjunto de prueba y de testeo en 70%,30%
xtrain, xtest, ytrain, ytest =train_test_split(x, #valores de los predictores
                                               y, #los valores del target
                                               test_size=0.3, #proporción para datos de testeo
                                               random_state=2021, #semilla
                                               stratify=y) #la variable de estratificación


# ### ESCALAMIENTO DE VARIABLES

# In[184]:


#Estandarización: Instancia StandardScaler
sc=StandardScaler()


# In[185]:


xtrain_std=sc.fit_transform(xtrain)
xtrain_std


# In[186]:


#Con lo aprendido de Xtrain debemos realizar la transformacion para eñ xtest
xtest_std=sc.transform(xtest)
xtest_std


# In[187]:


columnas=["porc_hogares_sin_medios","IDH ","alfabetismo","porc_2_NB","GINI"]


# In[188]:


df_std=pd.DataFrame(xtrain_std,
                   columns=columnas)
df_std.head()


# In[189]:


df_std.shape


# ## 1. ANÁLISIS DE COMPONENTES PRINCIPALES

# ### CONSTRUCCION DE LA MATRIZ DE CORRELACIÓN

# In[190]:


df_corr=df_std.corr(method="pearson")
df_corr


# ### Prueba de esferacidad de Bartlet

# In[191]:


n = df_std.shape[0] #número de observaciones
p = df_std.shape[1] #número de columnas
chi2 = -(n-1-(2*p+5)/6)*math.log(np.linalg.det(df_corr))
chi2


# In[192]:


ddl = p*(p-1)/2
ddl


# In[193]:


p= stats.chi2.pdf(chi2,ddl)
p


# COMO EN LA PRUEBA DE LA ESFERICIDAD DE BARTLET NOS SALE "0.0", POR LO TANTO, SI ES OPTIMO APLICAR EL MÉTODO DE ANÁLISIS DE COMPONENTES PRINCIPALES

# ### ANÁLISIS DE COMPONENTES PRINCIPALES

# In[194]:


pca=PCA()#Inicializamos en PCA
xtrain_pca=pca.fit_transform(xtrain_std)
VarianzaExplicada=pca.explained_variance_ratio_
VarianzaExplicada


# In[195]:


VarianzaAcomulada=np.cumsum(pca.explained_variance_ratio_)
VarianzaAcomulada


# ### GRÁFICA PCA

# In[196]:


with plt.style.context('dark_background'):
    plt.bar(range(1, 6), VarianzaExplicada,color=colors)
    plt.step(range(1, 6), VarianzaAcomulada, color="yellow")
plt.ylabel('Explicación de Variabilidad')
plt.xlabel('Componestes Principales')
plt.title("GRÁFICA PCA",weight='bold',color='red')
plt.show()


# ANALIZANDO LA GRÁFICA PCA PODEMOS OBSERVAR QUE LA MAYOR CANTIDAD DE VARIABLES SE ENCUENTRA EN LA PRIMERA COMPONENTE PERO PARA ESTAR SEGUROS USEMOS EL CRITERIO DE KAISER Y DEBEMOS OBSERVAR QUE VARIABLES DE  LA MATRIZ DE CORRELACION DE VARIANZA SEA MAYOR A UNO. AQUELLOS DATOS QUE CUMPLAN CON ESA CONDICION SERAN MIS **COMPONENTES PRINCIPALES**.

# ##### Criterio de Kaiser

# In[197]:


cov_mat = np.cov(xtrain_std.T)
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
print('\nEigenvalues \n%s' % eigen_vals)


# In[198]:


print("La cantidad de COMPONENTES PRINCIPALES ES: ",(eigen_vals>0.98).sum() )


# SEGÚN EL CRITERIO DE KAISER LA CANTIDAD DE COMPONESTES PRINCIPALES SON 3 YA QUE HABIAMOS PREDICHO PERO AHORA SI LO PODEMOS ASEGURAR

# In[199]:


print("El porcentaje de los datos tomados por LOS COMPONENTES PRINCIPALES ES:")
print((eigen_vals[0]/sum(eigen_vals))*100 + (eigen_vals[1]/sum(eigen_vals))*100 + (eigen_vals[2]/sum(eigen_vals))*100,"%")


# ### GENERANDO EL NÚMERO DE COMPONENTES PRINCIPALES

# In[200]:


pca = PCA(n_components=2) #n_components es el número de componentes que nos indicó Kaiser
#fit_transform:ajuste el modelo con X y la reducción de dimensionalidad en X.
x_std = pca.fit_transform(xtrain_std)


# In[201]:


df_x =pd.DataFrame(x_std)
df_x.columns = ['PC1','PC2']
df_x.head()


# In[202]:


df_y = pd.DataFrame(ytrain)
df_y.columns = ['departamento']
df_y.head()


# ##### Nuevo conjunto de datos

# In[203]:


df_rd = pd.concat([df_x, df_y], axis=1)
df_rd.head(10)


# In[204]:


df_rd.info()


# ## 2. ANÁLISIS FACTORIAL

# Las librerias para el Analisis Factorial son:

# In[205]:


pip install factor_analyzer


# In[206]:


import sys
from factor_analyzer import FactorAnalyzer #Ojo es necesario descargar el paquete y colocarlo en una de las direcciones del path
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.decomposition import FactorAnalysis


# #### PRUEBA DE ESFERICIDAD DE BARTLET

# In[207]:


chi_square_value,p_value=calculate_bartlett_sphericity(df_std)
chi_square_value, p_value


# #### KMO (Kaiser-Meyer-Olkin) 

# In[208]:


#KMO (Kaiser-Meyer-Olkin) Test
kmo_all,kmo_model=calculate_kmo(df_std)
kmo_model


# UNA VEZ ANALIZADOS LOS DATOS POR LA ESFERICIDAD DE BARTLET Y EL KMO PODEMOS CONCLUIR QUE SI PODEMOS UTILIZAR EL MÉTODO DE ANÁLISIS FACTORIAL

# In[209]:


factorial= FactorAnalyzer()#Creamos la instancia FactorAnalyzer()
factorial.fit(df_std)
# Check Eigenvalues
ev, v = factorial.get_eigenvalues()
ev.round(2)


# In[210]:


with plt.style.context('dark_background'):
    plt.scatter(range(1,df_std.shape[1]+1),ev,color=colors[2])
    plt.plot(range(1,df_std.shape[1]+1),ev,color="yellow")
plt.title('GRÁFICO DE SEDIMENTACIÓN',weight='bold',color='red')
plt.xlabel('Factores')
plt.ylabel('Autovalores')
plt.grid()
plt.show()


# In[211]:


factores=factorial.get_factor_variance()
factores


# In[212]:


factorial_factores = FactorAnalyzer(n_factors=2,rotation='varimax')
factorial_factores.fit(df_std)


# In[213]:


factorial_factores.loadings_


# In[214]:


#Comunalidades
factorial_factores.get_communalities()


# In[215]:


#Especifidades
factorial_factores.get_uniquenesses()


# In[216]:


xtrain_factorial=factorial_factores.fit_transform(xtrain_std)


# In[217]:


df_fact =pd.DataFrame(xtrain_factorial)
df_fact.columns = ['PC1','PC2']
df_fact.head(10)


# In[218]:


df_fact.shape


# In[219]:


df_y = pd.DataFrame(ytrain)
df_y.columns = ['departamento']
df_y.head(10)


# In[220]:


df_rd_fact = pd.concat([df_fact, df_y], axis=1)
df_rd_fact.head(10)


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




