# Tratamiento de datos
# ==============================================================================
import pandas as pd
import numpy as np
import random

# Gráficos
# ==============================================================================
import matplotlib.pyplot as plt
import matplotlib_venn as vplt
import seaborn as sns

# Preprocesado y modelado
# ==============================================================================
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from funciones import CargarPandasDataset, ModificarColumnasValor, BorrarColumnas, InsertarColumnaNueva

# Configuración warnings
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

def clasificacionAbuso(row):
    if row['P10_8_1'] == 1.0 or row['P10_8_2'] == 1.0 or row['P10_8_3'] == 1.0 or row['P10_8_4'] == 1.0 or row['P10_8_5'] == 1.0 or row['P10_8_6'] == 1.0 or row['P10_8_7'] == 1.0:
        return 1
    return 0

def clasificacionAtencion(row):
    if row['P10_8_8'] == 1.0 or row['P10_8_9'] == 1.0 or row['P10_8_10'] == 1.0 or row['P10_8_13'] == 2.0 or row['P10_8_14'] == 2.0:
        return 1
    return 0

ruta_parcial = "/conjunto_de_datos_endireh_2021_csv/"

#Cargar datos SECCION I. CARACTERÍSTICAS DE LA VIVIENDA Y HOGARES EN LA VIVIENDA
seccionI = CargarPandasDataset(ruta_parcial+"conjunto_de_datos_TVIV.csv")
### Modificar columnas Binarias
columnasOHE = [f'P1_4_{i}' for i in range(1,10)]
seccionI = ModificarColumnasValor(df=seccionI, cols=columnasOHE, valorR=2, valorN=0)
### Política de datos faltantes
seccionI = ModificarColumnasValor(df=seccionI, cols=['P1_9'], valorR=np.nan, valorN=1)
### Creación de nueva columna BIENES_DE_VIVIENDA
seccionI['BIENES_DE_VIVIENDA'] = seccionI[[f'P1_4_{i}' for i in range(1,10)]].T.sum()
### Borrar columnas
labels = ['CVE_ENT', 'NOM_ENT', 'NOM_MUN', 'COD_RES', 'P1_8', "VIV_SEL", "UPM", 'UPM_DIS', 'EST_DIS', 'FAC_VIV', 'CVE_MUN', 'ESTRATO']
labels.extend([F'P1_4_{i}' for i in range(1,10)])
seccionI = BorrarColumnas(df=seccionI, cols=labels)


#Cargar datos SECCION II. CARACTERÍSTICAS SOCIODEMOGRÁFICAS DE RESIDENTES DE LA VIVIENDA
seccionII = CargarPandasDataset(ruta_parcial+"conjunto_de_datos_TSDem.csv")
### Modificar columnas Binarias
columnasOHE = ['P2_8', 'P2_9', 'P2_11', 'P2_13']
seccionII = ModificarColumnasValor(df=seccionII, cols=columnasOHE, valorR=2, valorN=0)
### Política de datos faltantes
seccionII[seccionII.NIV>2] = ModificarColumnasValor(df=seccionII[seccionII.NIV>2], cols='P2_8', valorR=np.nan, valorN=1)
seccionII = ModificarColumnasValor(df=seccionII, cols='P2_8', valorR=9, valorN=np.nan)
### Discretizar P2_5 y P2_6
seccionII = ModificarColumnasValor(df=seccionII, cols=['P2_5'], valorR=range(31), valorN=1)
seccionII = ModificarColumnasValor(df=seccionII, cols=['P2_5'], valorR=96, valorN=2)
seccionII = ModificarColumnasValor(df=seccionII, cols=['P2_5'], valorR=97, valorN=3)
seccionII = ModificarColumnasValor(df=seccionII, cols=['P2_5'], valorR=98, valorN=np.nan)
seccionII = ModificarColumnasValor(df=seccionII, cols=['P2_6'], valorR=range(31), valorN=1)
seccionII = ModificarColumnasValor(df=seccionII, cols=['P2_6'], valorR=96, valorN=2)
seccionII = ModificarColumnasValor(df=seccionII, cols=['P2_6'], valorR=97, valorN=3)
seccionII = ModificarColumnasValor(df=seccionII, cols=['P2_6'], valorR=98, valorN=np.nan)
### Discretizar columna PAREN
seccionII = ModificarColumnasValor(df=seccionII, cols=['PAREN'], valorR=[5,6,7,8,9], valorN=4)
seccionII = ModificarColumnasValor(df=seccionII, cols=['PAREN'], valorR=10, valorN=5)
seccionII = ModificarColumnasValor(df=seccionII, cols=['PAREN'], valorR=11, valorN=6)
### Borrar columnas
labels = ['NOM_ENT', 'NOM_MUN', 'NOMBRE', 'N_REN', 'REN_MUJ_EL', 'REN_INF_AD', 'CODIGO', 'COD_M15', 'P2_16', "VIV_SEL", "HOGAR", "UPM", 'UPM_DIS', 'EST_DIS', 'ESTRATO', 'FAC_VIV', 'FAC_MUJ', 'CVE_MUN']
seccionII = BorrarColumnas(df=seccionII, cols=labels)


#Cargar datos SECCION IV. INGRESOS Y RECURSOS
seccionIV = CargarPandasDataset(ruta_parcial+"conjunto_de_datos_TB_SEC_IV.csv")
### Modificar columnas Binarias
columnasOHE = ['P4_1', 'P4_11']
columnasOHE.extend([f'P4_12_{i}' for i in range(1,8)])
columnasOHE.extend([f'P4_8_{i}' for i in range(1,8)])
seccionIV = ModificarColumnasValor(df=seccionIV, cols=columnasOHE, valorR=2, valorN=0)
### Política de datos faltantes
columnasNaN = ['P4_3', 'P4_5_1_AB', 'P4_6_AB']
seccionIV = ModificarColumnasValor(df=seccionIV, cols=columnasNaN, valorR=9, valorN=np.nan)
seccionIV = ModificarColumnasValor(df=seccionIV, cols=['P4_7_AB'], valorR=999998, valorN=np.nan)
seccionIV = ModificarColumnasValor(df=seccionIV, cols=['P4_7_AB'], valorR=999999, valorN=np.nan)
### Creación de nuevas columnas FUENTES_DE_DINERO y PROPIEDADES_DEL_HOGAR
seccionIV['FUENTES_DE_DINERO'] = seccionIV[[f'P4_8_{i}' for i in range(1,8)]].T.sum()
seccionIV['PROPIEDADES_DEL_HOGAR'] = seccionIV[[f'P4_12_{i}' for i in range(1,8)]].T.sum()
### Borrar columnas
labels = ['NOM_ENT', 'NOM_MUN', 'N_REN', 'N_REN_ESP', 'P4A_1', 'P4_4', 'P4_4_CVE',
         "VIV_SEL", "HOGAR", "UPM", 'UPM_DIS', 'EST_DIS', 'ESTRATO', 'FAC_VIV', 'FAC_MUJ', 'CVE_MUN']
labels.extend([f'P4_8_{i}' for i in range(1,8)])
labels.extend([f'P4_10_{i}_{j}' for i in [2,3] for j in [1,2,3]])
labels.extend([f'P4_12_{i}' for i in range(1,8)])
seccionIV = BorrarColumnas(df=seccionIV, cols=labels)


#Cargar datos SECCION X. ATENCIÓN OBSTÉTRICA
seccionX = CargarPandasDataset(ruta_parcial+"conjunto_de_datos_TB_SEC_X.csv")
### Preservar los datos de las embarazadas entre 2016 y 2021
seccionX = seccionX[(seccionX.P10_2 == 1)]
seccionX.drop(seccionX[((seccionX["P10_6ANIO"]<2016.0) | (seccionX["P10_6ANIO"]>2021.0))].index, axis=0, inplace=True)
### Discretizar años de parto
seccionX["P10_6ANIO"] = seccionX["P10_6ANIO"]-2015
#### Conservar solo los registros de hospitales públicos
seccionX.drop(seccionX[(seccionX.P10_7 > 5)].index, inplace = True)
### Política de datos faltantes
seccionX.loc[40712, 'P10_4_3'] = 1.0
### Creación de nuevas columnas 
#### SERVICIOS_MEDICOS_AFILIADA
seccionX['SERVICIOS_MEDICOS_AFILIADA'] = np.nan
for i in range(1,10):
    seccionX['SERVICIOS_MEDICOS_AFILIADA'].loc[seccionX[seccionX[f'P10_1_{i}']==1].index] = i-1
seccionX['SERVICIOS_MEDICOS_AFILIADA'].loc[seccionX[(seccionX['P10_1_1']==1) & (seccionX['P10_1_3']==1)].index] = 9
seccionX['SERVICIOS_MEDICOS_AFILIADA'].loc[seccionX[(seccionX['P10_1_1']==1) & (seccionX['P10_1_6']==1)].index] = 10
seccionX['SERVICIOS_MEDICOS_AFILIADA'].loc[seccionX[(seccionX['P10_1_1']==1) & (seccionX['P10_1_7']==1)].index] = 11
#### DONDE_CONSULTAS_PRENATALES
seccionX['DONDE_CONSULTAS_PRENATALES'] = np.nan
for i in range(11):
    seccionX['DONDE_CONSULTAS_PRENATALES'].loc[seccionX[seccionX[cols[i]]==1].index] = i
seccionX['DONDE_CONSULTAS_PRENATALES'].loc[seccionX[(seccionX['P10_5_01']==1) & (seccionX['P10_5_05']==1)].index] = 11
seccionX['DONDE_CONSULTAS_PRENATALES'].loc[seccionX[(seccionX['P10_5_02']==1) & (seccionX['P10_5_06']==1)].index] = 12
#### P10_8_abuso
seccionX_cls = InsertarColumnaNueva(df=seccionX_cls, nombreCol='P10_8_abuso', numeroCol=41, funcion=clasificacionAbuso)
#### P10_8_atencion
seccionX_cls = InsertarColumnaNueva(df=seccionX_cls, nombreCol='P10_8_atencion', numeroCol=42, funcion=clasificacionAtencion)
### Borrar columnas
labels = ['NOM_ENT', 'NOM_MUN', 'N_REN', 'P10_2', 'P10_8',
         "VIV_SEL", "HOGAR", "UPM", 'UPM_DIS', 'EST_DIS', 'FAC_VIV', 'FAC_MUJ', 'CVE_MUN', 'ESTRATO']
labels.extend([f'P10_1_{i}' for i in range(1,10)])
labels.extend([f'P10_5_0{i}' for i in range(1,10)])
labels.extend([f'P10_5_{i}' for i in range(10,12)])
labels.extend([f'P10_8_{i}' for i in range(1,16)])
seccionX_cls = BorrarColumnas(df=seccionX_cls, cols=labels)


# UNIR TODAS LAS SECCIONES
result = pd.merge(seccionI, seccionII, how="inner")
result = pd.merge(result, seccionIV, how="inner")
result = pd.merge(result, seccionX_cls, how="inner")
result.reset_index(drop=True, inplace=True) #reajustar el índice
## Borrar columnas
result = BorrarColumnas(df=result, cols=["ID_PER", 'ID_VIV', 'SEXO'])
## Quitar caracteres especiales
encoder = OrdinalEncoder()
result['DOMINIO'] = encoder.fit_transform(result[['DOMINIO']])
result['T_INSTRUM'] = encoder.fit_transform(result[['T_INSTRUM']])
## Normalizar columnas continuas
scaler = MinMaxScaler()
result['P1_3'] = scaler.fit_transform(result[['P1_3']])
result['P4_5_AB'] = scaler.fit_transform(result[['P4_5_AB']])
result['P4_7_AB'] = scaler.fit_transform(result[['P4_7_AB']])
## Discretizar edades
result['EDAD'] = result.EDAD+(result.P10_6ANIO-6)
for i in range(8):
    actual = i*5 + 10
    result.loc[result[(result["EDAD"]>=actual) & (result["EDAD"]<actual+5)].index, "EDAD"] = i
## Borrar columnas 
### Con muchos valores faltantes
columnas_nan_borrar = []
for col,nan in zip(result.columns,list(result.isnull().sum())):
    if nan > result.shape[0]/3:
        columnas_nan_borrar.append(col)
endireh = BorrarColumnas(result, columnas_nan_borrar)
### Que relfejan la misma información
endireh = BorrarColumnas(endireh, ['P2_13'])
## Renombrar columnas
renombrar_columnas = {'T_INSTRUM' : 'SITUACION_CONYUGAL', 'P1_1': 'MATERIAL_PISOS', 'P1_2': 'CUARTOS_DORMIR', 'P1_2_A': 'CUARTOS_TOTAL', 'P1_3': 'FOCOS',
           'P1_5': 'AGUA', 'P1_6': 'DRENAJE', 'P1_7': 'NUM_RESIDENTES', 'P1_9': 'NUM_HOGARES',
           
           'P2_5': 'RES_MADRE', 'P2_6': 'RES_PADRE', 'NIV': 'ESCOLARIDAD', 'GRA': 'GRADO_ESCOLAR', 'P2_8': 'ALFABETISMO', 
           'P2_9': 'ASISTENCIA_ESC', 'P2_10': 'PERT_INDIGENA', 'P2_11': 'LENG_INDIGENA', 
           
           'P4AB_1': 'VERIF_SITUACION_PAREJA', 'P4_1': 'ENTREVISTADA_TRABAJA', 'P4_3': 'PAREJA_TRABAJA', 'P4_5_AB': 'PAREJA_GANANCIAS',  
           'P4_5_1_AB': 'PAREJA_GANANCIAS_FRECUENCIA', 'P4_6_AB': 'PAREJA_APORTA_PARA_GASTO', 'P4_7_AB': 'PAREJA_CUANTO_APORTA_GASTO', 
           'P4_11': 'LIBERTAD_USAR_DINERO', 
           
           'P10_3': 'NUM_EMBARAZOS', 'P10_4_1': 'NACIO_VIV', 'P10_4_2': 'NACIO_MUERT', 'P10_4_3': 'ABORTO', 'P10_6ANIO': 'ANIO_PARTO',
           'P10_6MES': 'MES_PARTO', 'P10_7': 'DONDE_ATENDIO_PARTO'}
endireh.rename(columns = renombrar_columnas, inplace=True)


# Guardar datos
endireh.to_csv(f'data/endireh_nac.csv', index=False)

gk = endireh.groupby('DOMINIO')

for dom in endireh.DOMINIO.unique():
    X = gk.get_group(dom)
    X = X.drop(columns=['DOMINIO'], inplace=False)
    X.to_csv(f'data/endireh_dom_{int(dom)}.csv', index=False)