# Tarea-No.-3-MPSS



# 1.
A partir de los datos, encontrar la mejor curva de ajuste (modelo probabilístico) para las funciones de densidad marginales de X y Y.2.

La curva de mejor ajuste para ambos casos (función de densidad marginal de X y de Y) es la "Normal" o "Gaussiana".  Esto porque a la hora de graficar los resultados obtenidos del análisis se observan las siguuientes figuras:

![Función de Densidad Marginal de X](https://github.com/MateoOG/Tarea-No.-3-MPSS/blob/master/MarginalDeX.png)

![Función de Densidad Marginal de Y](https://github.com/MateoOG/Tarea-No.-3-MPSS/blob/master/MarginalDeY.png)


Y se puede afirmar con certeza que a la curva que más se asemejan es a la "Normal" o "Gaussiana".



# 2.
Asumir independencia de X y Y. Analíticamente, ¿cuál es entonces la expresión de la función de densidad conjunta que modela los datos?

En este caso, dado que se asumen independencia estadística de las variables X y Y. La función de densidad conjunta sería de la siguiente forma:

### f<sub>x,y</sub>(x,y) = f<sub>x</sub>(X)* f<sub>y</sub>(Y)
 
Donde:f<sub>x</sub>(X) es la función de densidad marginal de X y  f<sub>y</sub>(Y) es la función de densidad marginal de Y


# 3.
Hallar los valores de correlación, covarianza y coeficiente de correlación (Pearson) para los datos y explicar su significado:

Los parámetros de la función de densidad marginal de X (mu,sigma) son, respectivamente: [9.90484381, 3.29944286]
Los parámetros de la función de densidad marginal de y (mu,sigma) son, respectivamente: [15.0794609,   6.02693774]

La correlación es 149.4773100000001

La Covarianza es 0.06481000000000012

El coeficiente de correlación es 0.0032592154154949104



La correlación indica qué tanto se parece el comportamiento de una de las variables en comparación con la otra; si una decrece la otra también lo hace, y viceversa. Es importante mencionar que el hecho de que exista correlación entre variables no implica "causalidad".
Donde causalidad se entiende en palabras sencillas como: si pasó A entonces pasará B.

La covarianza nos dice qué tanto se parece el comportamiento de las variables tomando como base de comparación sus respectivas medias. Nos dice qué tan dependientes son una de la otra.

Y por último, el coeficiente de correlación demuesta que tanto depende linealmente una variable de la otra. Se diferencia de la covarianza, porque es independiente de la escala de medida de las variables.


# 4.
Graficar las funciones de densidad marginales (2D), la función de densidad conjunta (3D).


Al usar los parámetros obtenidos de la comparación entre los datos del archivo .CSV y el modelo Gaussiano para graficar la función de densidad marginal de las variables X y Y se obtuvieron los siguientes resultados:

![Función Ajustada de Densidad Marginal de X ](https://github.com/MateoOG/Tarea-No.-3-MPSS/blob/master/MarginalOptDeX.png)

![Función Ajustada de Densidad Marginal de Y](https://github.com/MateoOG/Tarea-No.-3-MPSS/blob/master/MarginalOptDeY.png)



y, finalmente se grafica la función de Densidad Conjunta de las variables bajo análisis usando estos mismo parámetros. En este proceso se obtuvo:



![Función de Densidad Conjunta de X y Y](https://github.com/MateoOG/Tarea-No.-3-MPSS/blob/master/ConjuntaXY.png)

