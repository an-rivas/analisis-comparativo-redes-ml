# Custom functions
from funciones import CargarPandasDatasetCategoricos

# Tratamiento de datos
import numpy as np
import pandas as pd
import time
from os.path import exists
from sklearn.preprocessing import OrdinalEncoder

# Preprocesado y modelado
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.inspection import permutation_importance

# Configuración warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def ImportanciaDeVariables(param_grid, X, y, tipo='class'):
    if tipo=='regress':
        param_grid['learning_rate'] = 0.1
        param_grid['max_depth'] = 5
        param_grid['eval_metric'] = 'rmse'
        param_grid['objective'] = 'reg:squarederror'
        model = XGBRegressor(**param_grid)
        #hacer fit
        model.fit(X,y)
        #imprimir mejor score
        print(f'rmse: {mean_squared_error(model.predict(X), y, squared=False)}')
    else:
        param_grid['learning_rate'] = 0.3
        param_grid['max_depth'] = 8
        if tipo=='bin':
            param_grid['objective'] = 'binary:logistic'
            param_grid['eval_metric'] = 'logloss'
        else:
            param_grid['objective'] = 'reg:logistic'
            param_grid['eval_metric'] = 'mlogloss'
        model = XGBClassifier(**param_grid)
        #hacer fit
        model.fit(X,y)
        #imprimir mejor score
        if tipo=='bin':
            print(f'f1: {f1_score(model.predict(X), y, average="binary")}')
        else:
            print(f'f1: {f1_score(model.predict(X), y, average="macro")}') 
    
    return permutation_importance(model, X, y, random_state=5, n_jobs = -1).importances_mean

param_grid = {# estos son pre-elegidos
              'n_estimators'       : 50,
              'random_state'       : 5,
              'missing'            : np.nan,
              'enable_categorical' : True,
              'use_label_encoder'  : False,
              'n_jobs'             : -1,
              
              # estos cambian dependiendo del problema
              'eval_metric'        : f1_score,  
              'learning_rate'      : 0.3,
              'max_depth'          : 5,
              
              # estos son obtenidos en búsqueda en malla estáticos a todos los problemas
              'grow_policy'        : 'depthwise',
              'tree_method'        : 'approx',
             }

path = 'ENDIREH-data-analysis/Análisis de datos/Violencia_obstetrica_2024v/data/'

columnas_continuas = ['FOCOS', 'PAREJA_GANANCIAS', 'PAREJA_CUANTO_APORTA_GASTO']
columnas_dfaltantes = ['RES_MADRE', 'RES_PADRE', 'VERIF_SITUACION_PAREJA', 'PAREJA_TRABAJA', 'PAREJA_GANANCIAS', 
                'PAREJA_GANANCIAS_FRECUENCIA', 'PAREJA_APORTA_PARA_GASTO', 'PAREJA_CUANTO_APORTA_GASTO']
columnas_binarias = ['ALFABETISMO', 'ASISTENCIA_ESC', 'LENG_INDIGENA', 'ENTREVISTADA_TRABAJA',
               'LIBERTAD_USAR_DINERO', 'P10_8_abuso', 'P10_8_atencion']

datasets = ['endireh_nac.csv', 'endireh_dom_0.csv', 'endireh_dom_1.csv', 'endireh_dom_2.csv']

print("Tiempo de inicio:", time.strftime("%H:%M:%S", time.localtime()), '\n')

for nombre in datasets: #POR CADA UNO DE LOS DATASETS
    print(f'por cada dataset {nombre}')
    endireh = CargarPandasDatasetCategoricos(path+nombre)
    
    archivo_PFI = 'pesos_'+'_'.join(nombre.split('_')[1:])
    
    if exists(path+archivo_PFI): # SI EL ARCHIVO DE PESOS YA EXISTE
        # solo se lee
        PFI = pd.read_csv(path+archivo_PFI, index_col='Unnamed: 0')
    else:
        # se crea y se guarda como vacio
        PFI = pd.DataFrame(index=endireh.columns)
        PFI.to_csv(path+archivo_PFI)
    
    for i,col in enumerate(endireh.columns): #POR CADA COLUMNA EN EL DATASET
        
        if col not in PFI.columns: # SI LA COLUMNA TODAVÍA NO EXISTE EN EL CSV DE PESOS PFI
            print(col, i) #print de control para saber que está trabajando
            
            #### SEPARAR EN X,y
            # SI ES DE LAS COLUMNAS CON VALORES NULOS, obtener los registros no nulos
            filas_seleccionadas = endireh[~endireh[col].isnull()] if col in columnas_dfaltantes else endireh
            # declarar la variable objetivo actual y sacarla de la matriz
            y = filas_seleccionadas[col].copy()
            X = filas_seleccionadas.drop(columns=col, inplace=False)
        
            # hacer la gradient bossting machine y obtener el feature importance
            if col in columnas_continuas:
                #regression
                fi = ImportanciaDeVariables(param_grid, X, y.astype('int64'), tipo='regress')
            elif col in columnas_binarias:
                #clasificacion binaria
                fi = ImportanciaDeVariables(param_grid, X, y.astype('bool'), tipo='bin')
            else:
                #clasificacion de multiples valores en una sola etiqueta
                #### ASEGURAR QUE <<y>> TENGA VALORES CONTINUOS
                unicos = sorted(y.unique()) # si es multiclass esta linea da problemas
                if unicos[-1] >= len(unicos): # SI FALTA ALGUN VALOR EN LA VARIABLE ACTUAL
                    mapeo = {valor: i for i,valor in enumerate(unicos)} # mapeo de los valores unicos a una secuencia continua
                    y = y.map(mapeo)
                # hacer que sea ordenada
                y = pd.Categorical(y, categories=range(len(unicos)), ordered=True)
                # clasificacion de multiples valores en una sola etiqueta
                f1, dicc = BusquedaEnMalla(param_grid, X, y, tipo='class')

            # insertar el valor nan a la columna actual
            fi = np.insert(fi, i, np.nan)

            # agregarlo como nueva columna 
            PFI[col] = fi

            # guardar el archivo
            PFI.to_csv(path+archivo_PFI)

            print("Tiempo actual:", time.strftime("%H:%M:%S", time.localtime()), '\n')
