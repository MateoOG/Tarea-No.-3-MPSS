# -*- coding: utf-8 -*-


# se importan las librerias necesarias
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats 

from scipy.optimize import curve_fit







# se crean los vectores de X y de Y
X_1 = np.arange(5,16,1)
Y_1 = np.arange(5,26,1)

# se crean dos listas para almacenar los valores discretos de las funciones de densidad marginales
listaX= []
listaY= []




Leerdatos = pd.read_csv('xy.csv') # se leen los datos del archivo que contiene las variables medidas
  
DF = pd.DataFrame(Leerdatos) # se convierten los datos en un DataFrame

CC = DF.values.tolist()# se pasa del DataFrame a lista




# se almacenan los valores de la PMF de X
for i in range(0,len(CC)):
    listaX.append((sum(CC[i])))
 
# se almacenan los valores de la PMF de Y  
listaY.append(DF['y5'].sum())
listaY.append(DF['y6'].sum())
listaY.append(DF['y7'].sum())
listaY.append(DF['y8'].sum())
listaY.append(DF['y9'].sum())
listaY.append(DF['y10'].sum())
listaY.append(DF['y11'].sum())
listaY.append(DF['y12'].sum())
listaY.append(DF['y13'].sum())
listaY.append(DF['y14'].sum())
listaY.append(DF['y15'].sum())
listaY.append(DF['y16'].sum())
listaY.append(DF['y17'].sum())
listaY.append(DF['y18'].sum())
listaY.append(DF['y19'].sum())
listaY.append(DF['y20'].sum())
listaY.append(DF['y21'].sum())
listaY.append(DF['y22'].sum())
listaY.append(DF['y23'].sum())
listaY.append(DF['y24'].sum())
listaY.append(DF['y25'].sum())






# se imprimen los datos para analizar la curva de mejor ajuste
plt.plot(X_1,listaX)
plt.title('Función de Densidad de Probabilidad')
plt.ylabel('probabilidad')
plt.xlabel('valores de X')
plt.savefig('MarginalDeX.png')
plt.show()



plt.plot(Y_1,listaY)
plt.title('Función de Densidad de Probabilidad')
plt.ylabel('probabilidad')
plt.xlabel('valores de Y')
plt.savefig('MarginalDeY.png')
plt.show()






# se comparan los datos obtenidos con el modelos normal para obtener los parámetros
def Gaussiana (x, mu, sigma):
    return 1/(np.sqrt(2*np.pi*sigma**2))*np.exp(-(x-mu)**2/(2*sigma**2))

param0, _ = curve_fit(Gaussiana,X_1,listaX)
param1, _ = curve_fit(Gaussiana,Y_1,listaY)

print('Los parámetros de la función de densidad marginal de X (mu,sigma) son, respectivamente:',param0)
print('Los parámetros de la función de densidad marginal de y (mu,sigma) son, respectivamente:',param1)


mux_1=10
sigmax_1=3.3

muy_1=15
sigmay_1=6




# Se calcula: covarianza, correlación y coeficiente de correlación
Leerdatos_1 = pd.read_csv('xyp.csv') # se leen los datos del archivo que contiene las X,Y y la probabilidad asociada a cada par

DF_1 = pd.DataFrame(Leerdatos_1)

CC_1 = DF_1.values.tolist()

Lista_C = []
Lista_C_1 = []

import operator
import functools

for i in range (0,len(CC_1)):
    Lista_C.append((functools.reduce(operator.mul, CC_1[i], 1)))
    
print('La correlación es' ,sum(Lista_C))


PromX =10

PromY =15


Lista = []

Lista_Covar=[]


a = 0

with open('xyp.csv') as datos:
    linea = datos.read().splitlines()
    
    for l in linea:
        linea=l.split(',')
        Lista.append([float(linea[0]), float(linea[1]), float(linea[2])])


        
for l in Lista:
    a = (l[0]-PromX)*(l[1]-PromY)*l[2]
    Lista_Covar.append(a)
    
    
print('La Covarianza es' ,sum(Lista_Covar)  )
    
   

VarianzaX=3.2994
VarianzaY=6.0269

Raw = (sum(Lista_Covar)/(VarianzaX*VarianzaY))

print('El coeficiente de correlación es', Raw)






# Se imprimen los gráficos de las densidades maginales usando los parámetros encontrados
plt.plot(X_1,stats.norm.pdf(X_1,mux_1,sigmax_1), 'DarkBlue') # FDP
plt.title('Probabilidad marginal de X')
plt.ylabel('probabilidad')
plt.xlabel('Valores d X')
plt.savefig('MarginalOptDeX.png')
plt.show()

plt.plot(Y_1,stats.norm.pdf(Y_1,muy_1,sigmay_1), 'pink') # FDP
plt.title('Probabilidad marginal de Y')
plt.ylabel('probabilidad')
plt.xlabel('valores de Y')
plt.savefig('MarginalOptDeY.png')
plt.show()



#Se imprime la función de densidad Conjunta

def Normal_Total_print(a,muX, sigmaX,b, muY, sigmaY):
    return (1/(np.sqrt(2*np.pi*sigmaX**2))*np.exp(-(a-muX)**2/(2*sigmaX**2)))*(1/(np.sqrt(2*np.pi*sigmaY**2))*np.exp(-(b-muY)**2/(2*sigmaY**2)))



X, Y = np.meshgrid(X_1,Y_1)


F = Normal_Total_print(X,mux_1,sigmax_1,Y,muy_1,sigmay_1)


from mpl_toolkits import mplot3d

fig=plt.figure()
eje=plt.axes(projection='3d')
eje.plot_surface(X,Y,F,rstride=1,cstride=1, cmap='Spectral_r',edgecolor='yellow')
plt.title('Función de Densidad Conjunta')
eje.set_xlabel('Valores de X')
eje.set_ylabel('Valores de Y')
eje.set_zlabel('Probabiliad Conjunta')
plt.savefig('ConjuntaXY.png')
plt.show()



