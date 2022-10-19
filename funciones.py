import pandas as pd
import copy

def CargarPandasDataset(ruta):
    """
    Carga un dataset de pandas a partir de un archivo csv con espacios vacios representados por un \r
    Recibe:
        ruta_parcial: string con la ruta parcial o absoluta al archivo csv.
    Devuelve:
        pandas dataset
    """
    return pd.read_csv(ruta, na_values=['\r'])

def CargarPandasDatasetCategoricos(ruta):
    """
    Carga el archivo csv del proyecto resultante del preprocesamiento de datos indicando qué columnas son categóricas y numéricas. 
    Recibe:
        ruta_parcial: string con la ruta parcial o absoluta al archivo csv.
    Devuelve:
        pandas dataset
    """
    """
    numericos = {'P1_2': pd.to_numeric,
                 'P1_2_A': pd.to_numeric,
                 'P9_3': pd.to_numeric,
                 'P9_6': pd.to_numeric,
                 'P1_3': pd.to_numeric,
                 'P1_7': pd.to_numeric,
                 'P2_9': pd.to_numeric,
                 'P2_11': pd.to_numeric,
                 'P2_12': pd.to_numeric,
                 'P2_13': pd.to_numeric,
                 'P9_8': pd.to_numeric}
    """
    numericos = ['P1_2','P1_2_A','FOCOS','P1_7','P1_9', 'P9_3','NACIO_VIV','NACIO_MUERT','ABORTO','P9_8']
    
    df_cat = CargarPandasDataset(ruta)
    df_num = df_cat.loc[:,numericos]
    df_cat = df_cat.astype('category')
    df_cat[numericos] = df_num
    
    #return pd.read_csv(ruta, dtype=object, converters=numericos)
    return df_cat

def GuardarDataset(df, name):
    """
    Guarda el dataset de pandas en la ruta indicada
    Recibe:
          df: dataset de pandas
        name: nombre con el que será guardado
    Devuelve:
        nada
    """
    df.to_csv(name, index=False)
    return

def ModificarColumnasValor(df, cols, valorR, valorN):
    """
    Modifica las columnas indicadas del dataframe de pandas donde un valor deba ser reemplazado por un valor nuevo
    Recibe:
            df: dataset de pandas
          cols: columnas a remplazar valores
        valorR: valor a remplazar
        valorN: valor nuevo
    Devuelve:
        pandas dataframe modificado
    """
    df_copy = copy.copy(df)
    df_copy[cols] = df[cols].replace(valorR, valorN)
    return df_copy

def BorrarColumnas(df, cols):
    """
    Devuelve un pandas dataframe sin las columnas indicadas 
    Recibe:
             df: dataset de pandas
           cols: columnas a remplazar valores
    Devuelve:
         pandas dataframe
    """
    df_new = df.drop(labels=cols, axis=1, inplace=False)
    return df_new

def InsertarColumnaNueva(df, nombreCol, numeroCol, funcion):
    """
    Inserta una columna nueva nombreCol, en la posicion numero col, en el dataset de pandas df, en base a la funcion 
    Recibe:
               df: dataset de pandas
        nombreCol: nombre de la columna nueva
        numeroCol: numero de columna donde se insertará
          funcion: funcion de Python
    Devuelve:
        pandas dataset
    """
    df_copy = copy.copy(df)
    df_copy.insert(numeroCol, nombreCol, df.apply(funcion, axis=1))
    return df_copy

def obtenerOhe(endireh):
    # Defino las columnas que no necesitan preprocesar a OHE (One Hot Encoding).
    ## columnas continuas
    columns = ["P1_2", "P1_2_A", 'P1_3', 'P1_7', 'P2_9', 'P2_11', 'P2_12', 'P2_13', "P9_3", "P9_6", "P9_8_11"]
    ## columnas categoricas que ya son OHE
    columns.extend([f'P1_4_{i}' for i in range(1,10)]) # bienes de vivienda
    columns.extend([F'P9_1_{i}' for i in range(1,11)]) # afiliacion a servicio de salud
    columns.extend([F'P9_4_{i}' for i in range(1,4)]) # afiliacion a servicio de salud
    columns.extend([F'P9_5_{i}' for i in range(1,13)]) # atencion obstetrica preparto

    # Aparto las columnas del dataset original.
    endireh_num = endireh[columns].copy()
    
    # Elimino las columnas del dataset
    endireh_cat = endireh.drop(columns=columns, axis=1, inplace=False)
    
    # Obtengo el OHE
    endireh_cat = pd.get_dummies(endireh_cat)
    
    # Concateno los conjuntos de datos OHE y continuos.
    endireh_ohe = pd.concat([endireh_cat, endireh_num], axis=1)
    
    return endireh_ohe
