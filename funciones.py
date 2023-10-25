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
    return pd.read_csv(ruta, na_values=['\r'], encoding='latin1')

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
    #return pd.read_csv(ruta, dtype=object, converters=numericos)
    """
    
    numericos = [ 'FOCOS', 'PAREJA_CUANTO_APORTA_GASTO', 'PAREJA_GANANCIAS']
    binarios = ['ALFABETISMO', 'ASISTENCIA_ESC', 'LENG_INDIGENA', 'ENTREVISTADA_TRABAJA', 
               'LIBERTAD_USAR_DINERO', 'P10_8_abuso', 'P10_8_atencion']
    
    df_cat = CargarPandasDataset(ruta)
    
    df_num = df_cat.loc[:,numericos]
    df_bin = df_cat.loc[:,binarios]
    
    df_bin = df_bin.astype('bool')
    df_cat = df_cat.astype('category')
    
    df_cat[numericos] = df_num
    df_cat[binarios]  = df_bin.values.tolist()
    
    return df_cat

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
