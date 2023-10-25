# Custom functions
from funciones import CargarPandasDatasetCategoricos

# Tratamiento de datos
"""
Versiones
numpy 1.23.3
pandas 1.5.0
xgboost 1.6.2
sklearn 1.1.2
Python 3.8.14
"""
import numpy as np
import pandas as pd
import time
from os.path import exists
from sklearn.preprocessing import OrdinalEncoder

# Preprocesado y modelado
import multiprocessing
from xgboost import XGBClassifier, XGBRegressor, DMatrix
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV,ParameterGrid
from sklearn.inspection import permutation_importance

# Configuración warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

def grid(param_grid, X, y, tipo='class', cv=5):
    #hacer grid search
    if tipo=='regress':
        param_grid['learning_rate'] = 0.3
        param_grid['tree_method'] = 'approx'
        param_grid['eval_metric'] = 'rmse'
        param_grid['objective'] = 'reg:squarederror'
        model = XGBRegressor(**param_grid)
        #hacer fit
        model.fit(X,y)
        #imprimir mejor score
        print(f'rmse: {mean_squared_error(model.predict(X), y, squared=False)}')
        return permutation_importance(model, X, y, random_state=5, n_jobs = -1).importances_mean
    else:
        if tipo=='bin':
            param_grid['learning_rate'] = 0.3
            param_grid['tree_method'] = 'approx'
            param_grid['objective'] = 'binary:logistic'
        else:
            param_grid['learning_rate'] = 0.1
            param_grid['tree_method'] = 'hist'
            param_grid['objective'] = 'multi:softmax'
        param_grid['eval_metric'] = f1_score
        model = XGBClassifier(**param_grid)
        #hacer fit
        model.fit(X,y)
        #imprimir mejor score
        if tipo=='bin':
            print(f'f1: {f1_score(model.predict(X), y, average="binary")}')
        else:
            print(f'f1: {f1_score(model.predict(X), y, average="macro")}') 
        return permutation_importance(model, X, y, random_state=5, n_jobs = -1).importances_mean

param_grid = {'n_estimators'       : 150,
              'max_depth'          : 10,
              'grow_policy'        : 'depthwise',
              'learning_rate'      : 0.1,
              'tree_method'        : 'approx',
              'n_jobs'             : -1,
              'random_state'       : 5,
              'missing'            : np.nan,
              'enable_categorical' : True,
              'eval_metric'        : f1_score,
              'use_label_encoder'  : False,
             }

path = 'data/'

columnas_regression = ['FOCOS', 'PAREJA_GANANCIAS', 'PAREJA_CUANTO_APORTA_GASTO']
columnas_nan = ['RES_MADRE', 'RES_PADRE', 'VERIF_SITUACION_PAREJA', 'PAREJA_TRABAJA', 'PAREJA_GANANCIAS', 
                'PAREJA_GANANCIAS_FRECUENCIA', 'PAREJA_APORTA_PARA_GASTO', 'PAREJA_CUANTO_APORTA_GASTO']
columnasBin = ['ALFABETISMO', 'ASISTENCIA_ESC', 'LENG_INDIGENA', 'ENTREVISTADA_TRABAJA', 'PAREJA_TRABAJA', 'PAREJA_APORTA_PARA_GASTO', 
               'LIBERTAD_USAR_DINERO', 'P10_8_abuso', 'P10_8_atencion']
#columnas_mult_class = ['BIENES_DE_VIVIENDA', 'FUENTES_DE_DINERO', 'PROPIEDADES_DEL_HOGAR', 'SERVICIOS_MEDICOS_AFILIADA', 'DONDE_CONSULTAS_PRENATALES']

datasets = ['endireh_nac.csv', 'endireh_dom_0.csv', 'endireh_dom_1.csv', 'endireh_dom_2.csv']

print("Tiempo de inicio:", time.strftime("%H:%M:%S", time.localtime()), '\n')

for nombre in datasets: #POR CADA UNO DE LOS DATASETS
    print(f'por cada dataset {nombre}')
    #print(nombre)
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
            
            #### SEPARAR EN X y
            if col in columnas_nan: # SI ES DE LAS COLUMNAS CON VALORES NULOS
                # obtener los registros no nulos
                selected_rows = endireh[~endireh[col].isnull()]
            else:
                selected_rows = endireh
            y = selected_rows[col].copy()
            X = selected_rows.drop(columns=col, inplace=False)
        
            #### ASEGURAR QUE <<y>> TENGA VALORES CONTINUOS
            unicos = sorted(y.unique()) # si es multiclass esta linea da problemas
            if unicos[-1] >= len(unicos): # SI FALTA ALGUN VALOR EN EL DOMINIO ACTUAL
                y = OrdinalEncoder().fit_transform(y.to_frame()).squeeze()
        
            # hacer la gradient bossting machine y obtener el feature importance
            if col in columnas_regression:
                #regression
                fi = grid(param_grid, X, y.astype('int64'), tipo='regress')
            elif col in columnasBin:
                #clasificacion binaria
                fi = grid(param_grid, X, y.astype('bool'), tipo='bin')
            else:
                #clasificacion de multiples valores en una sola etiqueta
                fi = grid(param_grid, X, y, tipo='class')

            # insertar el valor nan a la columna actual
            fi = np.insert(fi, i, np.nan)

            # agregarlo como nueva columna 
            PFI[col] = fi

            # guardar el archivo
            PFI.to_csv(path+archivo_PFI)

            print("Tiempo actual:", time.strftime("%H:%M:%S", time.localtime()), '\n')
