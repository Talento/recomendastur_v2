import pandas as pd
from datetime import datetime, timedelta
import ast
import requests
import time
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderUnavailable, GeocoderTimedOut, GeocoderServiceError
import re
import logging
from geopy.distance import geodesic
from sklearn.cluster import KMeans
import os
from dotenv import load_dotenv
import streamlit as st
import unicodedata

# Cargar las variables del archivo .env
load_dotenv()

# Acceder a la API key
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
####################################################################################################
###################################### FUNCIONES INPUTS ############################################
####################################################################################################
def solicitar_fecha(mensaje):
    """
    Obtener las fechas que introduce el usuario y comprobar que es correcta.
    
    Parámetros:
    - mensaje: cadena de texto con la fecha en formato "dd-mm-yyyy"

    Salida:
    - fecha: fecha en foormato fecha
    """
    while True:
        fecha_str = input(mensaje)
        try:
            fecha = datetime.strptime(fecha_str, "%d-%m-%Y")
            if fecha >= datetime.today():
                return fecha
            else:
                print("La fecha debe ser igual o posterior a hoy.")
        except ValueError:
            print("Formato incorrecto. Usa dd-mm-aaaa.")


####################################################################################################
###################################### FUNCIONES AUXILIARES ########################################
####################################################################################################
def obtener_coordenadas(localidad,cities, intentos=3, delay=1):
    """
    Obtener las coordenadas de una localidad con manejo de errores y reintentos.

    Parámetros:
    - localidad: str, nombre del lugar (por ejemplo "Gijón, España")
    - intentos: int, número máximo de reintentos si hay error de red
    - delay: int, segundos de espera entre reintentos

    Devuelve:
    - Tuple (latitud, longitud) o None si no se encuentra o falla la geolocalización
    """
    ###### Primero chekear si esta en el csv
    cities_dict=cities.set_index('name')[['lat', 'lon']].to_dict(orient='index')

    if localidad in cities_dict.keys():
        return cities_dict[localidad]['lat'],cities_dict[localidad]['lon']
    
    ##### Si no está, lo buscamos
    else:

        geolocalizador = Nominatim(user_agent="recomendastur_app", timeout=10)

        for intento in range(intentos):
            try:
                ubicacion = geolocalizador.geocode(localidad)
                if ubicacion:
                    return ubicacion.latitude, ubicacion.longitude
                else:
                    return None
            except (GeocoderUnavailable, GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"Intento {intento+1}/{intentos} fallido para '{localidad}': {e}")
                time.sleep(delay)
            except Exception as e:
                print(f"Error inesperado al obtener coordenadas para '{localidad}': {e}")
                return None

        print(f"No se pudo obtener coordenadas para '{localidad}' tras {intentos} intentos.")
        return None
####################################################################################################
def haversine_vectorizado(lat1, lon1, lat2_array, lon2_array):
    """
    Calcula la distancia (en km) entre un punto fijo (lat1, lon1) y arrays de puntos (lat2_array, lon2_array)
    usando la fórmula de Haversine.
    """
    R = 6371.0  # radio de la Tierra en km

    # Convertir grados a radianes
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2_array)
    lon2_rad = np.radians(lon2_array)

    # Diferencias
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Fórmula de Haversine
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance = R * c

    return distance
####################################################################################################
def calcular_distancia(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia entre dos puntos.
    
    Parámetros:
    - latitud y longitud de cada uno de los puntos

    Salida:
    - distancia en km entre los puntos
    """
    return geodesic((lat1, lon1), (lat2, lon2)).km
####################################################################################################
def extraer_fragmento(texto, max_len=256):
    """
    Extrae un fragmento de texto de hasta `max_len` caracteres,
    cortando por el último espacio antes de alcanzar el límite.
    Se añaden puntos suspensivos al final.

    Parámetros:
    - texto: cadena de texto
    - max_len: número máximo de caracteres a extraer (default: 256)

    Salida:
    - fragmento de texto con '...' al final
    """
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return ""
    texto = str(texto)
    if len(texto) <= max_len:
        return texto
    fragmento = texto[:max_len]
    ultimo_espacio = fragmento.rfind(" ")
    if ultimo_espacio != -1:
        fragmento = fragmento[:ultimo_espacio]
    return fragmento.strip() + "..."
####################################################################################################
def calcular_precio_persona(alojamientos, n_personas):
    """
    Calcular el precio medio por persona de un alojamiento. Considera unicamente las opciones en las que la capacidad del alojamiento se ajusta 
    a la solicitud.
    
    Parámetros:
    - alojamientos: dataframe con la información de los alojamientos
    - n_personas: número de personas de la solicitud

    Salida:
    - alojamientos: dataframe con los registros en los que se ha podido calcular un precio medio por persona.
    """

    def calcular_precio(fila):
        try:
            # Verificar que los datos sean válidos
            if not isinstance(fila["Número de personas"], str) or not isinstance(fila["Precio"], str):
                return None
            
            # Convertir a listas
            capacidad = np.array(ast.literal_eval(fila["Número de personas"]))
            precios = np.array(ast.literal_eval(fila["Precio"]))
            
            if capacidad.size == 0 or precios.size == 0:
                return None
            
            # Número de noches
            noches = fila["n_noches"] if pd.notna(fila["n_noches"]) else 1
            
            # Filtrar habitaciones disponibles para ese número de personas
            indices = np.where(capacidad == n_personas)[0]
            
            if indices.size > 0:
                return round(((precios[indices] / capacidad[indices]) / noches).mean(), 0)
            else:
                return None
        except Exception as e:
            return None
    # Aplicar la función a cada fila
    alojamientos["Precio_persona"] = alojamientos.apply(calcular_precio, axis=1)
    
    # Filtrar valores no nulos
    return alojamientos.dropna(subset=["Precio_persona"]).reset_index(drop=True)
####################################################################################################
def calcular_precio_medio_restaurantes(restaurantes):
    """
    Calcular el precio medio por persona de un restaurante o cafetería. 
    
    Parámetros:
    - restaurantes: dataframe con la información de los restaurantes y cafeterías

    Salida:
    - restaurantes_aux: dataframe con los registros en los que se ha podido calcular un precio medio por persona.
    """
    # Paso 1: Convertir la columna de precios a listas numéricas
    def convertir_a_lista(x):
        try:
            return np.array(ast.literal_eval(x), dtype=int) if isinstance(x, str) and x != 'X' else None
        except (ValueError, SyntaxError):
            return None  # Si hay un error en la conversión, devolver None
    
    restaurantes['Precio'] = restaurantes['Precio'].apply(convertir_a_lista)
    
    # Paso 2: Calcular el precio medio por persona
    def calcular_precio_medio(precios):
        if precios is None or len(precios) == 0:
            return None
        return round(precios.mean(), 0)

    restaurantes['Precio_persona'] = restaurantes['Precio'].apply(calcular_precio_medio)

    # Paso 3: Filtrar restaurantes válidos
    restaurantes_aux = restaurantes[restaurantes["Precio_persona"].notna()].reset_index(drop=True)
    
    return restaurantes_aux
####################################################################################################
def calcular_indice_recomendacion(df, alpha=0.5, beta=0.3, gamma=0.2):
    """
    Calcular un indice de recomendación que calcula valoracion, precio y número de opiniones.
    Es una media ponderada de los valores normalizados. 
    
    Parámetros:
    - df: dataframe con los datos sobre los que se quiere calcular el índice de recomendación.
    - alpha: peso de la valoración.
    - beta: peso del número de opiniones o comentarios.
    - gamma: peso del precio.

    Salida:
    - df_ordenado: df original con el indice de recomendación calculado y ordenado de mayor a menor.
    """
    # Normalizar la valoración y el precio por persona
    df['valoracion_normalizada'] = (df['Valoración'] - df['Valoración'].min()) / (df['Valoración'].max() - df['Valoración'].min())
    df['precio_normalizado'] = (df['Precio_persona'] - df['Precio_persona'].min()) / (df['Precio_persona'].max() - df['Precio_persona'].min())
    
    # Calcular el índice de recomendación
    df['indice_recomendacion'] = (alpha * df['valoracion_normalizada'] + 
                                  beta * (df['Votos'] / (df['Votos'].max() + 1)) - 
                                  gamma * df['precio_normalizado'])
    df_ordenado = df.sort_values(by='indice_recomendacion', ascending=False).reset_index(drop=True)
    return df_ordenado
###################################################################################################################
# ========================= CONFIGURACIÓN DE LA API =========================

def obtener_tiempo_y_distancia(np_lats_lons, modo="driving"):
    """
    Consulta la API de Google Distance Matrix para obtener la distancia y duración del trayecto.

    Args:
        np_lats_lons (array de arrays): array de tamaño N formado por arrays de dos elementos lon y lat [lon,lat]
        modo (str): Modo de transporte ('driving' o 'walking').

    Returns:
        tuple: Distancia y duración como cadenas de texto (ej. '3.4 km', '12 mins'), 
               o (None, None) si la API falla.
    """
    
    ori='|'.join([str(dat).replace('[','').replace(']','').replace(' ','') for i,dat in enumerate(np_lats_lons[:-1].tolist())])
    dest='|'.join([str(dat).replace('[','').replace(']','').replace(' ','') for i,dat in enumerate(np_lats_lons[1:].tolist())])

    url = (
        f"https://maps.googleapis.com/maps/api/distancematrix/json"
        f"?origins={ori}&destinations={dest}&mode={modo}&key={API_KEY}"
    )
    try:
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200 and data.get("rows") and data["rows"][0]["elements"][0]["status"] == "OK":
            distance=[dat["elements"][i]["distance"]["text"] for i,dat in enumerate(data['rows'])]
            duration=[dat["elements"][i]["duration"]["text"] for i,dat in enumerate(data['rows'])]
            return distance, duration
        else:
            # En lugar de print, usa logging.error o logging.warning
            logging.error(f"Respuesta inválida de la API de Google para origen. Estado: {response.status_code}, Datos: {data}")
            return None, None
    except Exception as e:
        # En lugar de print, usa logging.error
        logging.error(f"Error al obtener distancia: {e}", exc_info=True)
        return None, None

# --------------------------------------------------------------------------
# NUEVA FUNCIÓN: PROCESAR EL DATAFRAME DE RUTA (se añadirá a Functions_itinerary.py)
def procesar_ruta_con_distancias(ruta_np, coche_disponible):
    """
    Itera sobre un DataFrame de ruta para calcular distancia y duración entre puntos consecutivos
    utilizando la API de Google Maps. Añade estas columnas al array.
    """
    #if ruta_np==np.array([]):
    #    return ruta_np

    # El modo de transporte se define en base a si el coche está disponible
    modo = "driving" if coche_disponible == "S" else "walking"
    
    # El primer punto en la ruta no tiene un "origen" previo en la misma ruta para calcular el trayecto
    # Por lo tanto, el primer trayecto es nulo.
    distancias = [None]
    duraciones = [None]

    #cojo solo las coordenadas que estan en la columnas 1 y 2
    np_lats_lons=ruta_np[:,[1,2]] 
        
    # Llama a la función de la API con todas las consultas
    dist, dur = obtener_tiempo_y_distancia(np_lats_lons, modo=modo)
        
    distancias+=dist
    duraciones+=dur
    
    # Asegurarse de que las listas de distancias/duraciones coincidan con el tamaño del DataFrame
    # Esto maneja casos donde la ruta puede tener un solo punto o errores previos
    if len(distancias) < len(ruta_np):
        distancias.extend([None] * (len(ruta_np) - len(distancias)))
        duraciones.extend([None] * (len(ruta_np) - len(duraciones)))
    
    distancias_np = np.array(distancias).reshape(-1, 1)
    duraciones_np = np.array(duraciones).reshape(-1, 1)

    # Apilamos horizontalmente
    ruta_np = np.hstack([ruta_np, distancias_np])#distancias de los trayectos
    ruta_np = np.hstack([ruta_np, duraciones_np])#duracion d elos trayectos
    
    return ruta_np
#############################################################################################################################
def anotar_tramo(nombre, fila, modo_transporte):

    """['Nombre 0','Latitud 1 ','Longitud 2','Descripción 3 ','Precio_persona 4 ','Valoración 5']
    Genera una línea de texto descriptiva para un tramo del itinerario.

    Args:
        nombre (str): Tipo de lugar (Alojamiento, Comida, etc).
        fila (pd.Series): Fila actual del DataFrame con datos del lugar.
        anterior_fila (pd.Series or None): Fila anterior para calcular distancia si existe.
        modo_transporte (str): 'S' si se usa coche, 'N' si se camina.

    Returns:
        str: Línea de texto descriptiva del tramo, incluyendo distancia si corresponde.
    """
    texto = f"{nombre}: {fila[0]}"
    texto += f" - Descripción: {fila[3]}"

    if 'Precio_persona' in fila:
        texto += f" - Precio de {fila[4]}"
    if 'Valoración' in fila:
        texto += f" - Valoración de {fila[5]}"
    #texto += f" - Coordenadas de ({fila[1]}, {fila[2]})"

    if len(fila) > 10 and fila[9] and fila[10]:
        transporte = "Coche" if modo_transporte == 'S' else "Andando"
        texto += f" - ({fila[9]}, {fila[10]}, {transporte})"
    else:
        texto += " - (Distancia no disponible)"

    return texto
####################################################################################################
def generar_texto_ruta(ruta_np, zona, dia, numero_ruta, modo_transporte, precio_total, fecha):
    """
    Genera una lista de líneas de texto que describen una ruta completa del día.

    Args:
        ruta_df (pd.DataFrame): DataFrame con los lugares de la ruta.
        zona (str): Zona del viaje.
        dia (int): Número del día global.
        numero_ruta (int): Identificador de la ruta (1 o 2).
        modo_transporte (str): 'S' para coche, 'N' para andando.

    Returns:
        list: Lista de líneas de texto describiendo el itinerario del día.
    """
    secciones = ["Alojamiento", "Desayuno", "Lugar turístico", "Comida", "Lugar turístico", "Cena"]
    texto = [f"\nRUTA {numero_ruta} - Zona {zona}, Día {dia} (Fecha: {fecha.strftime('%Y-%m-%d')})"]

    for i in range(min(len(ruta_np), len(secciones))):
        nombre = secciones[i]
        fila = ruta_np[i]
        linea = anotar_tramo(nombre, fila, modo_transporte)
        texto.append(linea)
    texto.append(f"Precio diario de la ruta: {precio_total} €")

    if len(ruta_np) < len(secciones):
        texto.append(f"[Aviso] Ruta incompleta: solo hay {len(ruta_np)} de los {len(secciones)} tramos esperados.")

    return texto

####################################################################################################
###################################### FUNCIONES SELECCIÓN #########################################
####################################################################################################
####################################################################################################
def filtrar_por_zona(df, lat_zona, lon_zona, radio):
    """
    Filtrar un dataframe y obtener los establecimientos de una determinada zona
    
    Parámetros:
    - df: dataframe con los datos sobre los que se hace la selección.
    - lat_zona, lon_zona: coordenadas del punto central.
    - radio: radio (en km) al que se buscan establecimientos desde el punto central.

    Salida:
    - df_zona: dataframe con todos los establecimientos del area considerada.
    """
    if lat_zona is None or lon_zona is None:
        raise ValueError(f"No se pudieron obtener las coordenadas de la zona")
    
    # Calcular la distancia de cada alojamiento a la zona principal
    df['distancia'] = haversine_vectorizado(lat_zona, lon_zona, df['Latitud'].values, df['Longitud'].values) 

    #df['distancia'] = df.apply(lambda row: calcular_distancia(lat_zona, lon_zona, row['Latitud'], row['Longitud']), axis=1)

    # Filtrar alojamientos dentro del radio máximo
    df_zona = df[df['distancia'] <= radio]

    return df_zona.reset_index(drop=True)
####################################################################################################
def seleccionar_lugares_turisticos(lugares_turisticos, lat_zona, lon_zona, radio_max, dias):
    """
    Seleccionar los lugares turísticos correspondientes a una determinada zona 
    
    Parámetros:
    - lugares_turisticos: dataframe con los datos sobre los que se hace la selección.
    - lat_zona, lon_zona: coordenadas del punto central.
    - radio_max: radio (en km) al que se buscan lugares turísticos desde el punto central.
    - dias: número de días para los que hay que seleccionar lugares turísticos.

    Salida:
    - lugares_seleccionados: dataframe con 4*dias lugares turisticos, elegidos aleatoriomente entre los más cercanos.
    """
    
    if lat_zona is None or lon_zona is None:
        raise ValueError(f"No se pudieron obtener las coordenadas de la zona")
    
    #filtrar por radio
    lugares_filtrados=filtrar_por_zona(lugares_turisticos, lat_zona, lon_zona, radio_max)
    
    # Seleccionar aleatoriamente 4 lugares turísticos
    if len(lugares_filtrados) >= (4*dias):
        lugares_seleccionados = lugares_filtrados.sample(n=(4*dias), replace=False)
    else:
        lugares_seleccionados = lugares_filtrados  # Si hay menos de 4, seleccionar todos
    
    return lugares_seleccionados.reset_index(drop=True)
####################################################################################################
def seleccionar_alojamientos(alojamientos, lat_zona, lon_zona, dias, n_personas, radio = 5):
    """
    Seleccionar los alojamientos correspondientes a una determinada zona.
    
    Parámetros:
    - alojamientos: dataframe con los datos sobre los que se hace la selección.
    - lat_zona, lon_zona: coordenadas del punto central.
    - n_personas: número de personas para las que se busca alojamiento.
    - radio: radio (en km) al que se buscan lugares turísticos desde el punto central.

    Salida:
    - alojamientos_seleccionados: dataframe con 4 lugares turisticos, elegidos aleatoriomente entre los más cercanos y mejor valorados.
    """
    # Extraermos los alojamientos de la zona considerada
    alojamientos_aux = filtrar_por_zona(alojamientos, lat_zona, lon_zona, radio)

    # Calculamos el indice de recomendación de cada alojamiento y ordenamos el dataframe
    alojamientos_aux = calcular_precio_persona(alojamientos_aux, n_personas)
    alojamientos_aux = calcular_indice_recomendacion(alojamientos_aux)

    # Seleccionamos 4 alojamientos
    alojamientos_seleccion = alojamientos_aux.iloc[:50].sample(n=min(2, len(alojamientos_aux.iloc[:50]))).reset_index(drop=True)
    return alojamientos_seleccion
####################################################################################################
def seleccionar_restaurantes(restaurantes, lat_zona, lon_zona, dias, radio = 10):
    """
    Seleccionar los restaurantes y cafeterías correspondientes a una determinada zona 
    
    Parámetros:
    - restaurantes: dataframe con los datos sobre los que se hace la selección.
    - lat_zona, lon_zona: coordenadas del punto central.
    - dias: número de días para los que hay que seleccionar restaurantes y cafeterías.
    - radio: radio (en km) al que se buscan lrestaurantes y cafeterías desde el punto central.
    

    Salida:
    - cafeterias_seleccion: dataframe con 2*dias cafeterías, elegidos aleatoriomente entre los más cercanos y mejor valoradas.
    - restaurantes_seleccion: dataframe con 2*dias restaurantes, elegidos aleatoriomente entre los más cercanos y mejor valoradas.
    """
    restaurantes_aux = filtrar_por_zona(restaurantes, lat_zona, lon_zona, radio)
    restaurantes_aux = calcular_precio_medio_restaurantes(restaurantes_aux)
    restaurantes_aux = calcular_indice_recomendacion(restaurantes_aux)

    # Separamos entre cafeterías otros restaurantes
    filtro_cafe = restaurantes_aux["Nombre"].str.contains(r"(?:café|cafe|cafetería|cafeteria|kfe)", case=False, na=False)

    # Filtramos los nombres que cumplen con la condición
    cafeterias_aux = restaurantes_aux[filtro_cafe ].reset_index(drop = True)
    restaurantes_aux = restaurantes_aux[~filtro_cafe].reset_index(drop = True)

    cafeterias_seleccion = cafeterias_aux.iloc[:50].sample(n=min(2*dias, len(cafeterias_aux.iloc[:50]))).reset_index(drop=True)
    restaurantes_seleccion = restaurantes_aux.iloc[:50].sample(n=min(4*dias, len(restaurantes_aux.iloc[:50]))).reset_index(drop=True)
    
    return cafeterias_seleccion, restaurantes_seleccion
####################################################################################################
def agrupar_lugares_turisticos(lugares_turisticos, lugares_por_dia=4):
    """
    Agrupar los lugares turísticos en conjuntos de 'lugares_por_dia' elementos,
    minimizando la distancia dentro de cada grupo, calculando el punto medio de cada grupo 
    y ordenandolos por grupos.

    Parámetros:
    - lugares_turisticos: dataframe con lugares turisticos (importante disponer de 'Latitud' y 'Longitud').
    - lugares_por_dia: Número de lugares por grupo (por defecto 4).

    Retorna:
    - df_ordenado: dataframe ordenado por los grupos generados.
    - grupos: Lista de listas, donde cada sublista contiene los índices de los lugares en un grupo.
    - puntos_medios: Lista de tuplas con las coordenadas (latitud, longitud) del punto medio de cada grupo.
    """
    n_lugares = len(lugares_turisticos)
    
    if n_lugares < lugares_por_dia:
        raise ValueError(f"Se requieren al menos {lugares_por_dia} lugares para formar un grupo.")

    # Obtener coordenadas
    coordenadas = lugares_turisticos[['Latitud', 'Longitud']].values

    # Definir número de clusters aproximado
    n_clusters = max(1, n_lugares // lugares_por_dia)
    
    # Aplicar K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    lugares_turisticos['grupo'] = kmeans.fit_predict(coordenadas)

    # Agrupar los lugares por el cluster asignado
    grupos_temp = lugares_turisticos.groupby('grupo').apply(lambda x: list(x.index)).tolist()
    
    # Ajustar los grupos para que tengan tamaños homogéneos
    grupos = []
    sobrantes = []
    
    for grupo in grupos_temp:
        while len(grupo) > lugares_por_dia:
            sobrantes.append(grupo.pop())  # Extraer sobrantes
        grupos.append(grupo)

    # Redistribuir sobrantes a los grupos con menos elementos
    for i in range(len(grupos)):
        while len(grupos[i]) < lugares_por_dia and sobrantes:
            grupos[i].append(sobrantes.pop())

    # Calcular puntos medios de cada grupo
    puntos_medios = [
        (lugares_turisticos.loc[grupo, 'Latitud'].mean(), lugares_turisticos.loc[grupo, 'Longitud'].mean())
        for grupo in grupos
    ]
    
    # Reorganizar el DataFrame en orden de grupos
    indices_ordenados = [i for grupo in grupos for i in grupo]  # Aplanar la lista de grupos
    df_ordenado = lugares_turisticos.loc[indices_ordenados].reset_index(drop=True)

    return df_ordenado, grupos, puntos_medios
####################################################################################################
def seleccionar_datos_viaje(alojamientos, lugares_turisticos, restaurantes, cities, zona, dias, n_personas, radio, ninos):
    """
    Seleccionar alojamientos, lugares turísticos, restaurantes y cafeterías según los parámetros del viaje.

    Parámetros:
    - alojamientos: dataframe con alojamientos disponibles.
    - lugares_turisticos: dataframe con lugares turísticos.
    - restaurantes: dataframe con restaurantes disponibles.
    - zona: lista de zonas a visitar 
    - dias: lista de días que se estará en cada zona (referente al lugar o lugares de alojamiento).
    - n_personas: número de personas en el viaje.
    - radio: radio máximo de desplazamiento desde la zona de alojamiento.
    - ninos: "S" si hay niños en el viaje, "N" si no.

    Retorna:
    - alojamientos_seleccion: dataframe con los alojamientos seleccionados.
    - turistico_seleccion: dataframe con los lugares turísticos seleccionados.
    - cafeterias_seleccion: dataframe con las cafeterías seleccionadas.
    - restaurantes_seleccion: dataframe con los restaurantes seleccionados.
    """
    start=time.time()
    
    # Inicializar dataframes finales
    df_aloj = pd.DataFrame()
    df_tur = pd.DataFrame()
    df_caf = pd.DataFrame()
    df_res = pd.DataFrame()

    # Filtrar lugares turísticos si no hay niños
    if ninos == "S":
        lugares_turisticos = lugares_turisticos[lugares_turisticos["Tipo_lugar_turistico"] != "Bodegas, llagares y queserías"]
    # Recorrer cada zona
    for i in range(len(zona)):
        lat_zona, lon_zona = obtener_coordenadas(zona[i],cities)
        # Obtener alojamientos cercanos a la zona
        alojamientos_aux = seleccionar_alojamientos(alojamientos, lat_zona, lon_zona,dias[i], n_personas) # 2*dias alojamientos por zona
        
        # Obtener lugares turísticos cercanos
        turistico_aux = seleccionar_lugares_turisticos(lugares_turisticos, lat_zona, lon_zona, radio, dias[i])# 4*dias lugares turisticos por zona

        # Si se visitará la zona más de un día, agrupar en 4 lugares por día
        if dias[i] != 1:
            turistico_aux, grupos, puntos_medios = agrupar_lugares_turisticos(turistico_aux, lugares_por_dia=4)
            cafeterias_aux = pd.DataFrame()
            restaurantes_aux = pd.DataFrame()
            # Para cada grupo, seleccionar restaurantes y cafeterías cercanos al punto medio del grupo
            for j in range(len(grupos)):
                cafeterias_aux2, restaurantes_aux2 = seleccionar_restaurantes(
                    restaurantes, puntos_medios[j][0], puntos_medios[j][1], 1
                )
                cafeterias_aux = pd.concat([cafeterias_aux, cafeterias_aux2], ignore_index=True)
                restaurantes_aux = pd.concat([restaurantes_aux, restaurantes_aux2], ignore_index=True)

        else:
            cafeterias_aux, restaurantes_aux = seleccionar_restaurantes(restaurantes, lat_zona, lon_zona, 1)
        
        df_aloj = pd.concat([df_aloj, alojamientos_aux], ignore_index=True)
        df_tur = pd.concat([df_tur, turistico_aux], ignore_index=True)
        df_caf = pd.concat([df_caf, cafeterias_aux], ignore_index=True)
        df_res = pd.concat([df_res, restaurantes_aux], ignore_index=True)

    
    return df_aloj, df_tur, df_caf, df_res
####################################################################################################
def seleccionar_eventos(eventos, zona, fecha_inicio, fecha_fin):
    """
    Seleccionar eventos que coinciden con el viaje.

    Parámetros:
    - eventos: dataframe con los eventos que se desarrollan en Asturias durante el año.
    - zona: lista de zonas a visitar 
    - fecha_inici, fecha_fecha: fechas de inicio y fin del viaje.

    Retorna:
    - eventos_zona: dataframe con los eventos seleccionados para el mes y el lugar seleccionados.
    """
    
    mes_inicio = fecha_inicio.month
    mes_fin = fecha_fin.month
    meses = [0, mes_inicio, mes_fin]

    eventos_date = eventos[eventos["Mes numero"].isin(meses)]
    eventos_date["Lugar_norm"] = (
    eventos_date["Lugar "]
        .astype(str)
        .str.lower()
        .apply(lambda x: ''.join(
            c for c in unicodedata.normalize('NFD', x)
            if unicodedata.category(c) != 'Mn'
        ))
    )

    # Normalizar la lista 'zona'
    zona_norm = [
        ''.join(
            c for c in unicodedata.normalize('NFD', str(z).lower())
            if unicodedata.category(c) != 'Mn'
        )
        for z in zona
    ]

    # Filtrar sin importar mayúsculas ni tildes
    eventos_zona = eventos_date[eventos_date["Lugar_norm"].isin(zona_norm)]

    eventos_zona["Descripcion"] = (
        eventos_zona["Fiesta"].astype(str) + 
        " (" + eventos_zona["Lugar "].astype(str) + 
        ") - " + eventos_zona["Día"].astype(str)
    )

    return eventos_zona
######################################################################################################################
###################################### GENERAR ITINERARIO  ###########################################################
######################################################################################################################
def get_itinerary(aloj_sel,turi_sel,cafes_sel,rest_sel, zonas_ui,dias_zona_ui,coche, fecha_inicio):
# Extraer la primera frase de las descripciones
    aloj_sel['Primera_Frase'] = aloj_sel['Descripción'].apply(extraer_fragmento)
    turi_sel['Primera_Frase'] = turi_sel['Descripción'].apply(extraer_fragmento)
    cafes_sel['Primera_Frase'] = cafes_sel['Descripción'].apply(extraer_fragmento)
    rest_sel['Primera_Frase'] = rest_sel['Descripción'].apply(extraer_fragmento)

    itinerario_texto_lista = []
    indice_aloj = 0 # Índice para el alojamiento y cafeterías
    indice_cafe = 0
    indice_turi = 0 # Índice para los lugares turísticos y restaurantes
    dia_global = 1 # Para llevar el conteo de días a lo largo de todas las zonas
    fecha = fecha_inicio

    # Precios totales de las rutas
    suma_total1 = 0
    suma_total2 = 0

    #Vamos a procesar los df a arrays para gilizarlo todo
    #Las columnas que se van a usar son estas
    sel_cols=['Nombre','Latitud','Longitud','Primera_Frase','Precio_persona','Valoración']#puedes usar Primera_Frase o Descripción pero en la posición 3 (contando el 0)
    #como el df de lugares no tienen algunas, las rellenamos
    turi_sel['Precio_persona']='X'
    turi_sel['Valoración']='X'
    #escoger columnas
    aloj_sel=aloj_sel[sel_cols]
    cafes_sel=cafes_sel[sel_cols]
    rest_sel=rest_sel[sel_cols]
    turi_sel=turi_sel[sel_cols]
    #pasar a array
    aloj_sel_val=aloj_sel.values
    cafes_sel_val=cafes_sel.values
    rest_sel_val=rest_sel.values
    turi_sel_val=turi_sel.values
    ####################################################################################################################
    ################################# Generar itinerario por cada zona y días asignados ################################
    ####################################################################################################################
    for idx, zona_actual_iter in enumerate(zonas_ui):
        dias_en_zona_iter = dias_zona_ui[idx]
        # Asegurarse de tener suficientes datos para la zona y días
        # Asumimos que seleccionar_datos_viaje ya ha filtrado, pero aquí prevenimos IndexError
        num_aloj_esperados_zona = 2
        num_cafe_esperados_zona = 2 * dias_en_zona_iter # 2 alojamientos/cafeterias por día (ruta 1 y 2)
        num_rest_turi_esperados_zona = 4 * dias_en_zona_iter # 4 lugares turisticos/restaurantes por día (2 por ruta)
        # Slice de los arrays para la zona actual
        aloj_zona = aloj_sel_val[indice_aloj : indice_aloj + num_aloj_esperados_zona]
        cafe_zona = cafes_sel_val[indice_aloj : indice_aloj + num_cafe_esperados_zona]
        rest_zona = rest_sel_val[indice_turi : indice_turi + num_rest_turi_esperados_zona]
        turi_zona = turi_sel_val[indice_turi : indice_turi + num_rest_turi_esperados_zona]
        for i in range(dias_en_zona_iter):
            # Generación de la Ruta 1 (mañana/principal)
            # Se usan try-except para manejar casos donde no haya suficientes datos
            try:
                ruta1_val = np.concatenate([#se genera array [6,6]
                    aloj_zona[0:1], # Alojamiento del día
                    cafe_zona[2*i:2*i+1], # Desayuno
                    turi_zona[4*i:4*i+1], # Primer lugar turístico
                    rest_zona[4*i:4*i+1], # Comida
                    turi_zona[4*i+1:4*i+2], # Segundo lugar turístico
                    rest_zona[4*i+1:4*i+2]  # Cena
                ])
            except IndexError:
                st.warning(f"Advertencia: No se pudieron encontrar suficientes datos para la Ruta 1 del día {dia_global} en {zona_actual_iter}. El itinerario puede estar incompleto.")
                ruta1 = np.array([]) #crear array vacio para no fallar
            if not ruta1_val.size == 0:
                #Generar columnas a añadir al array
                dia_np = np.array([dia_global]*len(ruta1_val)).reshape(-1, 1)
                zona_np = np.array([zona_actual_iter]*len(ruta1_val)).reshape(-1, 1)
                rut_num_np = np.array([1]*len(ruta1_val)).reshape(-1, 1)
                # Apilamos horizontalmente
                ruta1_val = np.hstack([ruta1_val, dia_np])#dia
                ruta1_val = np.hstack([ruta1_val, zona_np])#zona
                ruta1_val = np.hstack([ruta1_val, rut_num_np])#Para identificar la ruta
                suma_dia1 = 0
                for bloque in ruta1_val:
                    if not bloque[4] == "X":
                        suma_total1 += bloque[4]
                        suma_dia1 += bloque[4]

                try:
                    ruta1_val = procesar_ruta_con_distancias(ruta1_val, coche)#obtener distancias, se añaden en las columnas [9,10]
                except Exception as e:
                    st.warning(f"Error al calcular distancias/tiempos para Ruta 1 del día {dia_global}: {e}")
                 #se obtiene el texto de la recomendación   
                itinerario_texto_lista.extend(generar_texto_ruta(ruta1_val, zona_actual_iter, dia_global, 1, coche, suma_dia1, fecha))
    #################### ###################### Generación de la Ruta 2 (tarde/alternativa)  #############################################
            try:
                ruta2_val = np.concatenate([
                    aloj_zona[1:2], # Alojamiento alternativo/mismo que el de la mañana
                    cafe_zona[2*i+1:2*i+2], # Desayuno alternativo
                    turi_zona[4*i+2:4*i+3], # Tercer lugar turístico
                    rest_zona[4*i+2:4*i+3], # Comida alternativa
                    turi_zona[4*i+3:4*i+4], # Cuarto lugar turístico
                    rest_zona[4*i+3:4*i+4]  # Cena alternativa
                ])
                
            except IndexError:
                st.warning(f"Advertencia: No se pudieron encontrar suficientes datos para la Ruta 2 del día {dia_global} en {zona_actual_iter}. El itinerario puede estar incompleto.")
                ruta2 = np.array([]) 
            if not ruta2_val.size==0:
                #Generar columnas a añadir al array
                dia_np = np.array([dia_global]*len(ruta1_val)).reshape(-1, 1)
                zona_np = np.array([zona_actual_iter]*len(ruta1_val)).reshape(-1, 1)
                rut_num_np = np.array([2]*len(ruta1_val)).reshape(-1, 1)
                # Apilamos horizontalmente
                ruta2_val = np.hstack([ruta2_val, dia_np])#dia
                ruta2_val = np.hstack([ruta2_val, zona_np])#zona
                ruta2_val = np.hstack([ruta2_val, rut_num_np])#Para identificar la ruta
                suma_dia2 = 0
                for bloque in ruta2_val:
                    if not bloque[4] == "X":
                        suma_total2 += bloque[4]
                        suma_dia2 += bloque[4]
                try:
                    ruta2_val = procesar_ruta_con_distancias(ruta2_val, coche)#obtener distancias, se añaden en las columnas [9,10]
                except Exception as e:
                    st.warning(f"Error al calcular distancias/tiempos para Ruta 2 del día {dia_global}: {e}")
                itinerario_texto_lista.extend(generar_texto_ruta(ruta2_val, zona_actual_iter, dia_global, 2, coche, suma_dia2, fecha))
                #ruta_np, zona, dia, numero_ruta, modo_transporte
            dia_global += 1 # Incrementa el contador de días para el siguiente día
            fecha = fecha + timedelta(days=1)

        # Actualizar índices para la siguiente zona
        indice_aloj += num_aloj_esperados_zona
        indice_cafe += num_cafe_esperados_zona
        indice_turi += num_rest_turi_esperados_zona

    itinerario_texto_lista.extend([f"\nPrecio aproximado total RUTA 1: {suma_total1}", f"Precio aproximado total RUTA 2: {suma_total2}"])   # , "\nEventos:" 

    return itinerario_texto_lista