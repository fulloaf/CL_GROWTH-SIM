# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# #!python3
"""

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

"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List


def main():

    df = pd.read_csv("lookup_table.csv")
    df.index = df.id
    # ['id', 'next', 'Especie', 'Zona', 'DensidadInicial', 'SiteIndex', 'Manejo', 'Condicion', 'α', 'β', 'γ']

    config = {
        "horizonte": 40,
        "rodales": 20,
        "edades": [0, 14],
        "has": [1, 10],
        "raleos": [7, 25],
        "podas": [7, 25],
        "cosechas": [15, 30],
    }

    def calc_biomasa(rodal: pd.Series, e: int) -> float:
        """calcular la biomasa para un rodal y una edad, >=0"""
        return max(rodal["α"] * e ** rodal["β"] + rodal["γ"], 0)

    def generar_rodal(idx=None, edades=config["edades"], has=config["has"]) -> pd.Series:
        """elegir/sortear un modelo de la tabla, sortear edad y hectareas"""
        # FIXME : comentar proximas 2 lineas
        # edades=config["edades"]
        # has=config["has"]
        if not idx:
            idx = np.random.choice(df.id)
        return pd.concat(
            (
                df.loc[idx],
                pd.Series({"edad": np.random.randint(*edades), "ha": np.random.randint(*has)}),
            )
        )

    def genera_plan_de_manejo(
        raleos=config["raleos"],
        podas=config["podas"],
        cosechas=config["cosechas"],
    ) -> pd.Series:
        """sortear ano de raleo, poda y cosecha. Debe ser coherente: raleo < cosecha y poda < cosecha"""
        # FIXME : comentar proximas 3 lineas
        # raleos=config["raleos"]
        # podas=config["podas"]
        # cosechas=config["cosechas"]
        cosecha = np.random.randint(*cosechas)
        if cosecha < podas[1]:
            podas[1] = cosecha - 1
        if cosecha < raleos[1]:
            raleos[1] = cosecha - 1
        return pd.Series(
            {
                "raleo": np.random.randint(*raleos),
                "poda": np.random.randint(*podas),
                "cosecha": np.random.randint(*cosechas),
            }
        )

    def simula_rodal_plan(
        rodal=generar_rodal(), plan_mnjo=genera_plan_de_manejo(), horizonte=config["horizonte"]
    ) -> List[pd.Series]:
        """a partir de un rodal y un plan de manejo, simula la biomasa"""
        # FIXME : comentar proximas 3 lineas
        # horizonte = config["horizonte"]
        # rodal = generar_rodal()
        # plan_mnjo = genera_plan_de_manejo()
        rodal_plan = pd.concat((rodal, plan_mnjo))

        assert plan_mnjo.raleo < plan_mnjo.cosecha < horizonte

        # FIXME : comentar proximas
        # np.isnan(rodal.next)
        if np.isnan(rodal.next):
            rodal_plan.raleo = -1
            return (
                rodal.ha
                * pd.Series(
                    [calc_biomasa(rodal, rodal.edad + i) for i in range(rodal_plan.cosecha)]
                    + [0 for i in range(rodal_plan.cosecha, horizonte)]
                ),
                rodal_plan,
            )

        next_rodal = df.loc[rodal.next]
        # print(f"{plan_mnjo.raleo=}", f"{plan_mnjo.cosecha=}", f"{horizonte=}")
        return (
            rodal.ha
            * pd.Series(
                [calc_biomasa(rodal, rodal.edad + i) for i in range(rodal_plan.raleo)]
                + [calc_biomasa(next_rodal, rodal.edad + i) for i in range(rodal_plan.raleo, rodal_plan.cosecha)]
                + [0 for i in range(rodal_plan.cosecha, horizonte)]
            ),
            rodal_plan,
        )

    def simula_tabla(horizonte=config["horizonte"]):
        # FIXME : comentar proxima linea
        # horizonte = config["horizonte"]

        # para cada modelo, calcular biomasa hasta horizonte, retorna filas
        df_all = df.apply(lambda row: pd.Series([calc_biomasa(row, e) for e in range(horizonte)]), axis=1)
        # transponer, una columna un modelo
        df_alt = df_all.T

        # graficar
        names = ["Especie", "Zona", "DensidadInicial", "SiteIndex", "Manejo", "Condicion"]
        axes = df_alt.plot()
        box = axes.get_position()
        axes.set_position([box.x0, box.y0, box.width * 0.5, box.y0 + box.height])
        # legend_labels = [str(list(df.set_index('id').loc[col][names].to_dict().values())) for col in df_alt.columns]
        legend_labels = [df.loc[col][names].to_dict() for col in df_alt.columns]
        plt.legend(legend_labels, loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig("tabla.png")
        # plt.show()

    def simula_bosque(num_rodales=config["rodales"], horizonte=config["horizonte"]):
        num_rodales = config["rodales"]

        df_bm = pd.DataFrame()
        df_bo = pd.DataFrame()
        for i in range(num_rodales):
            rodal_id = f"rodal_{i}"
            rodal = generar_rodal(idx=np.random.choice(df.id[20:34]), edades=config["edades"], has=config["has"])
            # DE MOMENTO SOLO GENERA RODALES CON PINO COMO ESPECIE CON ESE ATRIBUTO EN idx.

            plan_mnjo = genera_plan_de_manejo()
            #       biomasa, rodal_plan = simula_rodal_plan(rodal, plan_mnjo)

            # Envolver la llamada a simula_rodal_plan() en un bucle while para manejar AssertionError
            while True:
                try:
                    biomasa, rodal_plan = simula_rodal_plan(rodal, plan_mnjo)
                    break  # Si no se produce AssertionError, salimos del bucle
                except AssertionError:
                    print("AssertionError: plan_mnjo.raleo < plan_mnjo.cosecha < horizonte. Intentando de nuevo...")
                    plan_mnjo = genera_plan_de_manejo()  # Generar un nuevo plan de manejo

            # get info de zona, site index y especie
            zona = df.loc[rodal.id, "Zona"]
            site_index = df.loc[rodal.id, "SiteIndex"]
            especie = df.loc[rodal.id, "Especie"]
            manejo = df.loc[rodal.id, "Manejo"]

            # Crear una serie de edades para cada período del horizonte
            edades_simuladas = pd.Series(range(rodal.edad, rodal.edad + horizonte))

            # Crear un DataFrame para el rodal que contenga la biomasa y la edad
            rodal_df = pd.DataFrame(
                {
                    "horizonte": range(horizonte),
                    "rodal_id": [rodal_id] * horizonte,
                    "Zona": [zona] * horizonte,
                    "SiteIndex": [site_index] * horizonte,
                    "Especie": [especie] * horizonte,
                    "Manejo": [manejo] * horizonte,
                    "Biomasa": biomasa,
                    "Edad": edades_simuladas,
                }
            )

            # Agregar el DataFrame del rodal al DataFrame global de biomasa
            df_bm = pd.concat([df_bm, rodal_df], ignore_index=True)

            # Agregar el plan de manejo al DataFrame global de bosque
            rodal_plan["Zona"] = zona
            rodal_plan["SiteIndex"] = site_index
            rodal_plan["Especie"] = especie
            rodal_plan["Manejo"] = manejo
            df_bo = pd.concat(
                (
                    df_bo,
                    rodal_plan.to_frame().T[
                        [
                            "id",
                            "next",
                            "edad",
                            "ha",
                            "raleo",
                            "poda",
                            "cosecha",
                            "Zona",
                            "SiteIndex",
                            "Especie",
                            "Manejo",
                        ]
                    ],
                ),
                axis=0,
            )

        # ordenar las columnas a mi pinta para calcular las metricas nuevas adicionales a la biomasa total aerea
        df_bm = df_bm[["horizonte", "rodal_id", "Zona", "SiteIndex", "Especie", "Edad", "Manejo", "Biomasa"]]

        df_bm.to_csv("biomasa.csv")
        df_bo.to_csv("forest.csv")

        return df_bm

    def getDict4Calibration(calibration_df: pd.DataFrame()) -> (dict(), dict()):

        age = {}
        nonFustPortion = {}

        for index, row in calibration_df.iterrows():
            key = (row["Especie"], row["Zona"], row["SI"], row["Tipo de manejo"])

            # Agregar la edad al diccionario age
            if key not in age:
                age[key] = []
            age[key].append(row["Edad (años)"])

            # Agregar la proporción al diccionario nonFustPortion
            if key not in nonFustPortion:
                nonFustPortion[key] = []
            total_no_fustal = pd.to_numeric(row["Total no fustal"], errors="coerce")
            total = pd.to_numeric(row["Total"], errors="coerce")
            if pd.notna(total_no_fustal) and pd.notna(total):
                nonFustPortion[key].append(total_no_fustal / total)
            else:
                # Si 'Total no fustal' o 'Total' no pueden ser convertidos a números,
                # se remueve la key de ambos diccionarios
                if key in age:
                    del age[key]
                if key in nonFustPortion:
                    del nonFustPortion[key]

        return age, nonFustPortion

    def NonFustBiomPortionEstFunc(
        age_to_stimate: int(), ObservedAge: list(), ObservedNonFustPortion: list()
    ) -> float():
        """
        La porción de biomasa aérea no arbórea se estima a través de una función que ajusta
        los datos observados siguiendo una curva de tendencia potencial. La función recibe la edad del
        rodal, los datos de campo de la edad y la porción no fustal aérea, y proporciona una porción estimada de
        biomasa aérea no arbórea para el rodal.
        """

        # Tomar el logaritmo de los datos
        log_age = np.log(ObservedAge)
        log_nonFustPortion = np.log(ObservedNonFustPortion)

        # Ajuste lineal de los datos transformados
        coefficients = np.polyfit(log_age, log_nonFustPortion, 1)
        b = coefficients[0]
        log_a = coefficients[1]

        # Convertir log_a de nuevo a a
        a = np.exp(log_a)

        # Calcular el resultado
        try:
            result = a * age_to_stimate**b
        except ZeroDivisionError:
            result = 1  # como a menor años es casi 100% no fustal, y log de cero es indef (se cae), le ponemos esta excepcion.

        # Asegurar que el resultado esté dentro del rango válido (entre 0 y 1)
        if result > 1:
            result = 1
        elif result < 0:
            result = 0

        return result

    def calcular_biomasa_no_fust(row):
        """Estima la Biomasa total no fustal de cada rodal"""
        especie = row["Especie"]
        zona = row["Zona"]
        site_index = row["SiteIndex"]
        manejo = row["Manejo"]
        edad = row["Edad"]

        # Calcular la biomasa no fustal utilizando la función definida anteriormente
        biomasa_no_fust = (
            NonFustBiomPortionEstFunc(
                edad, age[(especie, zona, site_index, manejo)], nonFustPortion[(especie, zona, site_index, manejo)]
            )
            * row["Biomasa"]
        )

        return biomasa_no_fust

    def calcular_biomasa_fust(row):
        """Estima la Biomasa fustal de cada rodal"""
        BioTotal = row["Biomasa"]
        bioNoFust = row["biomasaNoFust"]

        bioFust = BioTotal - bioNoFust

        return bioFust

    # ConvertCMFactor=0.9*(2.5/3)*2
    def CubicMetersEstFunct(row):
        """Estima los metros cubicos de biomasa fustal de cada rodal"""
        bioFust = row["biomasaFust"]
        ConvertCMFactor = 0.9 * (2.5 / 3) * 2

        return bioFust * ConvertCMFactor

    # ConvertLeafFactor=1/3
    def LeafBiomEstFunct(row):
        """Estima la Biomasa de las hojas de cada rodal"""
        bioNoFust = row["biomasaNoFust"]
        ConvertLeafFactor = 1 / 3

        return bioNoFust * ConvertLeafFactor

    def estimador_altura(age_to_stimate, EDAD_Curvas, ALTURA_Curvas):
        """Define la función estimadora de la altura de cada rodal a partir de un ajuste polinomial de grado dos"""

        coefficients = np.polyfit(EDAD_Curvas, ALTURA_Curvas, 2)

        # Los coeficientes del polinomio
        a, b, c = coefficients

        #  print(f"La ecuación del polinomio es: y = {a}x^2 + {b}x + {c}")

        return a * (age_to_stimate**2) + b * age_to_stimate + c

    def calcular_altura(df_curvas: pd.DataFrame(), df_SimForest: pd.DataFrame()) -> pd.DataFrame():
        """Estima la altura de cada rodal a partir de función estimador_altura()"""

        EDAD_Curvas = list(df_curvas.columns[3:11])
        EDAD_Curvas = sorted(EDAD_Curvas)

        height_SI_curves = {}
        for index, row in df_SIheightCurves.iterrows():
            key = (row["Especie"], row["Zona"], row["SI"])
            value = row[2:10].tolist()
            value = sorted(value)
            height_SI_curves[key] = value

        altura_estimada = []

        for i in range(len(df_SimForest)):
            especie = df_SimForest["Especie"][i]
            zona = df_SimForest["Zona"][i]
            site_index = df_SimForest["SiteIndex"][i]
            edad = df_SimForest["Edad"][i]

            # Obtener la curva de altura correspondiente desde el diccionario
            key = (especie, zona, site_index)
            curva_altura = height_SI_curves.get(key)

            if curva_altura is not None:
                # Calcular la altura estimada utilizando la función estimador_altura
                estimated_height = estimador_altura(edad, EDAD_Curvas, curva_altura)
                altura_estimada.append(max(0, estimated_height))

            else:
                # Si no se encuentra la curva de altura correspondiente, asignar un valor nulo
                altura_estimada.append(np.nan)

        # Añadir la lista de alturas estimadas como una nueva columna al DataFrame
        df_SimForest["AlturaEstimada"] = altura_estimada

        # Actualizar la altura estimada a cero cuando la biomasa es cero
        df_SimForest.loc[df_SimForest["Biomasa"] == 0, "AlturaEstimada"] = 0

        return df_SimForest

    def Calcula_Prof_copa(row):
        """Estima la profundidad de la copa a partir de la altura estimada"""

        biomasa = row["Biomasa"]
        AlturaEst = row["AlturaEstimada"]
        prof_Copa_factor1 = 0.44
        prof_Copa_factor2 = 3.5

        if AlturaEst == 0 or biomasa == 0:
            return 0
        else:
            return AlturaEst * prof_Copa_factor1 + prof_Copa_factor2

    def calcular_CBH(row):
        """Estima el CBH (en metros) de cada rodal"""
        # Obtener los parámetros necesarios para la función
        AlturaEstimada = row["AlturaEstimada"]
        ProfundidadCopa = row["ProfundidadCopa"]

        CBH = AlturaEstimada - ProfundidadCopa

        if CBH <= 0:
            return 0
        else:
            return CBH

    def calcular_CBD(row):
        """Estima el CBD (kg/m3) para cada rodal"""
        # Obtener los parámetros necesarios para la función
        biomasaHojas = row["biomasaHojas"]
        ProfundidadCopa = row["ProfundidadCopa"]

        try:
            # Calcular CBD
            CBD = (biomasaHojas * 1000) / (ProfundidadCopa * 10000)
        except ZeroDivisionError:
            CBD = 0

        return CBD

    def calcular_ProfundidadCopaPoda(row):
        """Estima profundidad de copa con poda para cada rodal"""
        # Obtener los parámetros necesarios para la función
        alturaEstimada = row["AlturaEstimada"]
        profundidadPodaFactor = 0.44

        return alturaEstimada * profundidadPodaFactor

    def calcular_CBHconPoda(row):
        """Estima el CBH con poda (en metros) para cada rodal"""
        # Obtener los parámetros necesarios para la función
        AlturaEstimada = row["AlturaEstimada"]
        ProfundidadCopaconPoda = row["ProfundidadCopaPoda"]

        # Calcular la biomasa no fustal utilizando la función definida anteriormente
        CBH_conPoda = AlturaEstimada - ProfundidadCopaconPoda

        return CBH_conPoda

    def calcular_CBDconPoda(row):
        """Estima el CBD (kg/m3) con poda para cada rodal"""
        # Obtener los parámetros necesarios para la función
        biomasaHojas = row["biomasaHojas"]
        ProfundidadCopaconPoda = row["ProfundidadCopaPoda"]

        try:
            # Calcular CBD
            CBD_conPoda = (biomasaHojas * 1000) / (ProfundidadCopaconPoda * 10000)
        except ZeroDivisionError:
            CBD_conPoda = 0

        return CBD_conPoda

    df_SimForest = simula_bosque(num_rodales=config["rodales"])

    # Leer el archivo para calibrar los estimadores. NA estaba en el archivo pero pandas lo leia como NaN, asi que
    # saco de la misma lista para que sea considerado como un string.

    cal_df = pd.read_excel(
        "CalibrationData.xlsx",
        sheet_name="data",
        keep_default_na=False,
        na_values=[
            "#N/A",
            "#N/A N/A",
            "#NA",
            "-1.#IND",
            "-1.#QNAN",
            "-NaN",
            "-nan",
            "1.#IND",
            "1.#QNAN",
            "N/A",
            "NULL",
            "NaN",
            "n/a",
            "nan",
            "null",
        ],
    )

    # age,nonFustPortion=getDict4Calibration(cal_df)

    df_SIheightCurves = pd.read_excel("SI_heightCurves.xlsx", sheet_name="Curves")

    age, nonFustPortion = getDict4Calibration(cal_df)

    df_SimForest["biomasaNoFust"] = df_SimForest.apply(calcular_biomasa_no_fust, axis=1)
    df_SimForest["biomasaFust"] = df_SimForest.apply(calcular_biomasa_fust, axis=1)
    df_SimForest["mc_bioFust"] = df_SimForest.apply(CubicMetersEstFunct, axis=1)
    df_SimForest["biomasaHojas"] = df_SimForest.apply(LeafBiomEstFunct, axis=1)
    df_SimForest = calcular_altura(df_SIheightCurves, df_SimForest)
    df_SimForest["ProfundidadCopa"] = df_SimForest.apply(Calcula_Prof_copa, axis=1)
    df_SimForest["CBH"] = df_SimForest.apply(calcular_CBH, axis=1)
    df_SimForest["CBD"] = df_SimForest.apply(calcular_CBD, axis=1)
    df_SimForest["ProfundidadCopaPoda"] = df_SimForest.apply(calcular_ProfundidadCopaPoda, axis=1)
    df_SimForest["CBH_conPoda"] = df_SimForest.apply(calcular_CBHconPoda, axis=1)
    df_SimForest["CBD_conPoda"] = df_SimForest.apply(calcular_CBDconPoda, axis=1)
    df_SimForest.to_excel("FullGrowthSimMetrics.xlsx", index=False)


if __name__ == "__main__":
    main()

# %%
main()

# %%
