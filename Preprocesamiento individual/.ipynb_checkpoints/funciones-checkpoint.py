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
    numericos = {'P1_2': pd.to_numeric,
                 'P1_2_A': pd.to_numeric,
                 'P9_3': pd.to_numeric,
                 'P9_6': pd.to_numeric}
    
    return pd.read_csv(ruta, dtype=object, converters=numericos)

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
