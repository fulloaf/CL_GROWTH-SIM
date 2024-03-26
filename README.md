# CL_GROWTH-SIM

@authors: Felipe Ulloa-Fierro & Fernando Badilla Véliz
v0.2


Simulador de crecimiento usando el archivo lookup_table.csv que resume la información descrita en "Modelos de predicción 
de biomasa a nivel de rodal en plantaciones de Eucalyptus globulus y Pinus radiata en Zona centro sur de Chile" 
sobre modelos de crecimiento con un cambio de manejo.

***Nota: simulaciones con información completa sólo para especies tipo pino.

Funciones:

* Creación de rodal: ha, edad actual, un modelo [Zona, densidad inicial, Índice de Sitio, Manejo, Condición y especie (pino o eucalipto)].
* Creación de plan de manejo: raleo, poda, cosecha.
* Generar biomasa: a partir de un rodal, plan y horizonte; calcular biomasa total aerea cambiando de modelo si hay raleo.
* Generar bosque: para cierto numero de rodales, generar biomasa y plan de manejo. output 2 archivos.csv:
    - Forest.csv: columnas: rodal, plan de manejo, filas: rodal_id_1..n
    - biomasa.csv: columnas: horizonte,rodal_id,Zona,SiteIndex,Especie,Edad,Manejo,Biomasa 
                      filas: anos_1..horizonte, rodal_id_1..n, escalar Zona, escalar SiteIndex, 
                             edadInicial...edadfinal (de acuerdo al horizonte), Tipo de Manejo, biomasa_de_rodal_id_1..n

* Obtener datos de calibración para métricas: age:dict(), nonFustPortion:dict() se obtienen a partir de datos de campo
  provistos y recopilados en el archivo excel CalibrationData.xlsx.
* Obtener función para estimar porción de biomasa no fustal: la misma se modela como una función potencial 
          de acuerdo a ajuste de tendencia obtenido sobre los datos observados.
* Estimador de la biomasa no fustal de cada rodal.
* Estimador de la biomasa fustal de cada rodal.
* Estimador de los metros cúbicos de biomasa fustal de cada rodal.
* Estimador de la biomasa de las hojas de cada rodal.
* Obtener función para estimar la altura de un rodal: la misma se modela como una función polinomial de grado 2, 
  de acuerdo a ajuste de tendencia sobre las curvas de indice de sitio según grupo zonal (***De momento tenemos sólo 
  las curvas para pino) descritas en el documento del Instituto Forestal y recopiladas en el archivo excel SI_heightCurves.xlsx.
* Estimador de la altura a partir de la función anterior, para cada rodal.
* Estimador de la profundidad de la copa a partir de la altura estimada de cada rodal.
* Estimador del CBH (en metros) de cada rodal.
* Estimador del CBD (en kg/m3) de cada rodal.
* Estimador de la profundidad de copa con poda para cada rodal.
* Estimador del CBH con poda (en metros) de cada rodal.
* Estimador del CBD (kg/m3) con poda para cada rodal.

Main Output: un archivo excel llamado forestFullGrowthSimMetrics.xlsx que junta la información de biomasa.csv 
             con las métricas estimadas para los rodales simulados.
