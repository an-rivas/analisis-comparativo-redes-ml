# Custom functions
from funciones import CargarPandasDatasetCategoricos

"""
Versiones
numpy 1.23.3
pandas 1.5.0
xgboost 1.6.2
sklearn 1.1.2
Python 3.8.14
"""
# Tratamiento de datos
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import OrdinalEncoder

# Preprocesado y modelado
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV,ParameterGrid

# Configuración warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

XGBRegressor


def BusquedaEnMalla(param_grid, X, y, tipo='class', cv=5):
    #hacer grid search
    if tipo=='regress':
        grid = GridSearchCV(XGBRegressor(), param_grid, cv=cv, scoring='neg_root_mean_squared_error', verbose=1, error_score="raise", n_jobs=-1)
    else:
        if  tipo=='bin':
            scoring = 'f1'
        else:
            scoring = 'f1_micro'
        grid = GridSearchCV(XGBClassifier(), param_grid, cv=cv, scoring=scoring, verbose=1, error_score="raise", n_jobs=-1)
    
    #hacer fit
    grid.fit(X, y)
    #devolver mejor score y mejores parametros
    return grid.best_score_, grid.best_params_

param_grid_r = ParameterGrid(
                            {
                             'n_estimators'       : [[50]],
                             'max_depth'          : [range(5,10)],
                             'grow_policy'        : [['depthwise', 'lossguide']],
                             'learning_rate'      : [[0.1, 0.2, 0.3]],
                             'tree_method'        : [['approx', 'hist']],
                             'random_state'       : [[5]],
                             'missing'            : [[np.nan]],
                             'enable_categorical' : [[True]],
                             'eval_metric'        : [['rmse']],
                             'use_label_encoder'  : [[False]],
                             'objective'          : [['reg:squarederror']],
                            }
                        )
param_grid = ParameterGrid(
                            {
                             'n_estimators'       : [[50]],
                             'max_depth'          : [range(5,10)],
                             'grow_policy'        : [['depthwise', 'lossguide']],
                             'learning_rate'      : [[0.1, 0.2, 0.3]],
                             'tree_method'        : [['approx']],
                             'random_state'       : [[5]],
                             'missing'            : [[np.nan]],
                             'enable_categorical' : [[True]],
                             'eval_metric'        : [[f1_score]], #'mlogloss'
                             'use_label_encoder'  : [[False]],
                             'objective'          : [['reg:logistic']],
                            }
                        )
    
path = 'ENDIREH-data-analysis/Análisis de datos/Violencia_obstetrica_2021_backup/data/'

columnas_continuas = ['FOCOS', 'PAREJA_GANANCIAS', 'PAREJA_CUANTO_APORTA_GASTO']
columnas_dfaltantes = ['RES_MADRE', 'RES_PADRE', 'VERIF_SITUACION_PAREJA', 
                'PAREJA_TRABAJA', 'PAREJA_GANANCIAS',
                'PAREJA_GANANCIAS_FRECUENCIA', 'PAREJA_APORTA_PARA_GASTO',
                'PAREJA_CUANTO_APORTA_GASTO']
columnas_binarias = ['ALFABETISMO', 'ASISTENCIA_ESC', 'LENG_INDIGENA', 
               'ENTREVISTADA_TRABAJA', 'LIBERTAD_USAR_DINERO', 
               'P10_8_abuso', 'P10_8_atencion']
#columnas_mult_class = ['BIENES_DE_VIVIENDA', 'FUENTES_DE_DINERO', 'PROPIEDADES_DEL_HOGAR', 'SERVICIOS_MEDICOS_AFILIADA', 'DONDE_CONSULTAS_PRENATALES']

nombre = 'endireh_nac.csv'

endireh = CargarPandasDatasetCategoricos(path+nombre)

#Se ignoran basado en no chingar los datos como ya estan establecidos
columnas_ignorar = ['CUARTOS_DORMIR', 'NUM_EMBARAZOS', 'NACIO_VIV', 'ABORTO']

parametros = ['tiempo', 'metric', 'grow_policy', 'learning_rate', 'tree_method', 'max_depth']
archivo_param = 'parameters_r1.csv'
param = pd.read_csv(path+archivo_param, index_col='Unnamed: 0')
aux = {p:[] for p in parametros}

print("Tiempo de inicio:", time.strftime("%H:%M:%S", time.localtime()), '\n')
for i,col in enumerate(endireh.columns): #POR CADA COLUMNA EN EL DATASET
    print(col, i) #print de control para saber que está trabajando
    
    if col not in param.columns and col not in columnas_ignorar:
        #### SEPARAR EN X y
        # SI ES DE LAS COLUMNAS CON VALORES NULOS, obtener los registros no nulos
        filas_seleccionadas = endireh[~endireh[col].isnull()] if col in columnas_dfaltantes else endireh
        # declarar la variable objetivo actual y sacarla de la matriz
        y = filas_seleccionadas[col].copy()
        X = filas_seleccionadas.drop(columns=col, inplace=False)

        print(X.shape, y.shape)
        tiempo = time.time()
        # hacer la búsqueda grid y obtener el feature importance
        if col in columnas_continuas:
            #regression
            f1, dicc = BusquedaEnMalla(param_grid_r, X, y.astype('int64'), tipo='regress')
        elif col in columnas_binarias:
            #clasificacion binaria
            f1, dicc = BusquedaEnMalla(param_grid, X, y.astype('bool'), tipo='bin')
        else:
            #clasificacion de multiples valores en una sola etiqueta
            #### ASEGURAR QUE <<y>> TENGA VALORES CONTINUOS
            unicos = sorted(y.unique()) # si es multiclass esta linea da problemas
            if unicos[-1] >= len(unicos): # SI FALTA ALGUN VALOR EN LA VARIABLE ACTUAL
                y = pd.Series(OrdinalEncoder().fit_transform(y.to_frame()).squeeze())
            #clasificacion de multiples valores en una sola etiqueta
            f1, dicc = BusquedaEnMalla(param_grid, X, y.astype('category'), tipo='class')

        aux['metric'] = f1
        aux['tiempo'] = time.time() - tiempo
        for k,v in dicc.items():
            if k in parametros:
                aux[k] = v
        param[col] = aux.values()

        print(aux)

        # guardar el archivo
        param.to_csv(path+archivo_param, index=True)

    print("Tiempo actual:", time.strftime("%H:%M:%S", time.localtime()), '\n')

"""
##obtener un solo valor para las columnas de multiple clasificacion
for col in columnas_mult_class:
aux = PFI.loc[desgloce_columnas_mult_class[col],:].median()
PFI.append(pd.Series(data=aux, name='_'.join(col.split('_')[:2])), ignore_index=False)
PFI.drop(index=desgloce_columnas_mult_class[col], inplace=False)
"""
