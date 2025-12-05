# itinerary_app_copy.py
import streamlit as st
import pandas as pd
from datetime import datetime
from Scripts.Functions_itinerary import (
    obtener_coordenadas,
    seleccionar_datos_viaje,
    get_itinerary,
    seleccionar_eventos
)

from Scripts.functions_llm import load_llm, LLM_output

import re
import torch
import warnings
import sys
import time

# Suprime advertencias, aunque Streamlit puede mostrarlas en su propio log
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

# ============================== CORRECCIÓN PARA EL ERROR DE TORCH.CLASSES ==============================
# Necesario para evitar un error de PyTorch en algunos entornos, especialmente si no hay CUDA
if hasattr(torch, 'classes') and hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = []
# ============================== CARGA DE DATOS ==============================
pipe=load_llm(remote=False)

@st.cache_data
def load_data():
    """
    Carga los datasets de alojamientos, restaurantes y puntos turísticos.
    Usa st.cache_data para evitar recargar los datos en cada interacción, mejorando el rendimiento.
    """
    try:
        # Asegúrate de que las rutas son correctas relative al script de Streamlit
        alojamientos = pd.read_csv("./Scripts/Definitive_data/accommodation_processed.csv")
        restaurantes = pd.read_csv("./Scripts/Definitive_data/restaurant_processed_cleaned.csv")
        turistico = pd.read_csv("./Scripts/Definitive_data/tourist_processed.csv")
        cities = pd.read_csv("./Scripts/Definitive_data/cities_coords.csv")
        events = pd.read_excel("./Scripts/Definitive_data/Fiestas_Interes_Turistico.xlsx", engine='openpyxl')

        return alojamientos, restaurantes, turistico,cities, events
    except FileNotFoundError:
        st.error("Error: Asegúrate de que los archivos CSV estén en la ruta './Scripts/Definitive_data/'.")
        st.stop() # Detiene la ejecución de la aplicación si los archivos no se encuentran

alojamientos, restaurantes, turistico, cities, events = load_data()

# Inicializar variables de estado de sesión si no existen
# Esto es fundamental en Streamlit para mantener los valores entre reruns de la aplicación
if 'texto_itinerario' not in st.session_state:
    st.session_state.texto_itinerario = None
if 'llm_reformatted_itinerario' not in st.session_state:
    st.session_state.llm_reformatted_itinerario = None
if 'show_download_original' not in st.session_state:
    st.session_state.show_download_original = False
if 'show_download_llm' not in st.session_state:
    st.session_state.show_download_llm = False
if 'execute_llm_model' not in st.session_state:
    st.session_state.execute_llm_model = False
if 'executed_response' not in st.session_state:
    st.session_state.executed_response = False

# ============================== INPUTS USUARIO (Barra Lateral) ==============================

st.sidebar.image("Logo-recomendastur.png", use_container_width=True)
st.sidebar.image("Logo-talento-blanco.png", use_container_width=True)


st.sidebar.title("Parámetros del Viaje")

# Selección de fechas
fecha_inicio = st.sidebar.date_input("Fecha de inicio", value=datetime.now())
fecha_fin = st.sidebar.date_input("Fecha de fin", value=fecha_inicio)

# Validación de fechas
if fecha_fin < fecha_inicio:
    st.sidebar.error("La fecha de fin debe ser posterior o igual a la de inicio.")
    # No se detiene la app completamente, pero se deshabilita el botón de generar
    dias = 0 # Para que el botón de generar se deshabilite
else:
    dias = (fecha_fin - fecha_inicio).days + 1

# Tipo y número de personas
tipo_viaje = st.sidebar.selectbox("Tipo de viaje", ["Solo", "Pareja", "Familia", "Amigos"])
if tipo_viaje == "Solo":
    n_personas = 1
    ninos = "No"
elif tipo_viaje == "Pareja":
    n_personas = 2
    ninos = "No"
else: # Para "Familia" o "Amigos"
    n_personas = st.sidebar.number_input("Número de personas", min_value=1, value=3)
    ninos = "No" # Valor por defecto

    if tipo_viaje == "Familia": # Solo preguntar por niños si es Familia
        ninos = st.sidebar.radio("¿Viajan niños?", ["Sí", "No"], index=1) # index=1 para 'N' por defecto

# Opción de coche y radio de búsqueda
coche = st.sidebar.radio("¿Dispones de coche?", ["Sí", "No"], index=0) # index=0 para 'S' por defecto
if coche == "Sí":
    radio = st.sidebar.number_input("Radio máximo de búsqueda (km)", min_value=5.0, value=30.0, help="Distancia máxima desde el alojamiento para buscar puntos de interés.")
else:
    radio = 5.0 # Radio más pequeño si no hay coche, asumiendo movilidad a pie/transporte público.

#######################################################################################################
################### Configuración de zonas del viaje (única o múltiples) ##############################
#######################################################################################################
zona_unica_checkbox_ui = st.sidebar.checkbox("¿Zona única para todo el viaje?", value=True)
zonas_ui = []
dias_zona_ui = []
zona_valida_flag = True # Flag para controlar si las zonas introducidas son válidas

if zona_unica_checkbox_ui:
    zona_val_ui = st.sidebar.text_input("Zona principal del viaje (ej: Gijon, Oviedo)", value="Gijon")
    if zona_val_ui:
        if obtener_coordenadas(zona_val_ui,cities):
            zonas_ui = [zona_val_ui]
            dias_zona_ui = [dias]
        else:
            st.sidebar.warning(f"Zona '{zona_val_ui}' no válida. Inténtalo de nuevo. Ej: Gijon, Oviedo, Aviles, Llanes, Cangas de Onis.")
            zona_valida_flag = False
    else:
        st.sidebar.info("Por favor, introduce una zona principal.")
        zona_valida_flag = False
else: # Múltiples zonas
    st.sidebar.markdown("### Distribución de días por zona")
    temp_zonas_ui = []
    temp_dias_zona_ui = []
    dias_ya_asignados_tracker = 0
    
    # Creamos un contenedor para los inputs de zonas dinámicos
    zonas_multi_container = st.sidebar.container()

    # Usamos un bucle para añadir campos de zona/días dinámicamente
    # Permitimos al usuario añadir hasta 'dias' zonas, o un máximo razonable como 7 si hay muchos días
    num_max_zonas_input = max(1, min(dias, 7)) # Mínimo 1, máximo 7 o los días totales

    for i in range(num_max_zonas_input):
        if dias_ya_asignados_tracker >= dias:
            st.sidebar.success("✅ Todos los días ya han sido asignados a zonas.")
            break
        with zonas_multi_container.expander(f"Configurar Zona {i+1}", expanded=(i==0 and dias_ya_asignados_tracker < dias)):
            zona_actual_multi_ui = st.text_input(f"Nombre de la zona {i+1}", key=f"zona_multi_name_{i}")
            
            if zona_actual_multi_ui:
                if obtener_coordenadas(zona_actual_multi_ui,cities):
                    max_dias_para_esta_zona = dias - dias_ya_asignados_tracker
                    if max_dias_para_esta_zona > 0:
                        n_dias_multi_ui = st.number_input(f"Días en {zona_actual_multi_ui}",
                                                         min_value=1,
                                                         max_value=max_dias_para_esta_zona,
                                                         value=min(1, max_dias_para_esta_zona), # Valor inicial sensato
                                                         key=f"dias_multi_num_{i}")
                        temp_zonas_ui.append(zona_actual_multi_ui)
                        temp_dias_zona_ui.append(n_dias_multi_ui)
                        dias_ya_asignados_tracker += n_dias_multi_ui
                    else:
                        st.info("Ya has asignado todos los días disponibles.")
                        break # Salir del bucle si ya no quedan días por asignar
                else:
                    st.warning(f"Zona '{zona_actual_multi_ui}' no válida. Omite o corrige.")
                    zona_valida_flag = False # Marcar como inválido si una zona no es válida
            elif i == 0 and dias > 0: # Si no se ha introducido ninguna zona para el primer slot
                 st.info("Introduce la primera zona de tu viaje.")
                 zona_valida_flag = False
    
    zonas_ui = temp_zonas_ui
    dias_zona_ui = temp_dias_zona_ui

    if zonas_ui and sum(dias_zona_ui) != dias:
        st.sidebar.error(f"La suma de días en zonas ({sum(dias_zona_ui)}) no coincide con los días totales del viaje ({dias}).")
        zona_valida_flag = False
    elif not zonas_ui and dias > 0:
        if not zona_unica_checkbox_ui: # Solo si el checkbox de zona única está desmarcado y no hay zonas
            st.sidebar.info("Por favor, define las zonas y los días para cada una.")
            zona_valida_flag = False


########################################################################################################################################
####################################### BOTÓN PRINCIPAL PARA GENERAR ITINERARIO ########################################################
########################################################################################################################################

# El botón se habilita solo si la suma de días coincide y las zonas son válidas
can_generate = (dias > 0 and zonas_ui and dias_zona_ui and sum(dias_zona_ui) == dias and zona_valida_flag)

if st.sidebar.button("📍 Generar Itinerario", disabled=not can_generate):
    if not can_generate:
        st.error("Por favor, completa correctamente todos los parámetros del viaje en la barra lateral.")
        # No se detiene, solo muestra el error
    else:
        with st.spinner("Generando itinerario... Esto puede tardar unos segundos."):
            start=time.time()
            # Llama a la función de selección de datos
            aloj_sel, turi_sel, cafes_sel, rest_sel = seleccionar_datos_viaje(
                alojamientos, turistico, restaurantes,cities,
                zonas_ui, dias_zona_ui,
                n_personas, radio, ninos
            )

            itinerario_texto_lista=get_itinerary(aloj_sel,turi_sel,cafes_sel,rest_sel,zonas_ui,dias_zona_ui,coche, fecha_inicio)

            # Añadir eventos al itinerario
            eventos_sel = seleccionar_eventos(events, zonas_ui, fecha_inicio, fecha_fin)
            print("Prueba\n", eventos_sel)

            if eventos_sel.empty:
                itinerario_texto_lista.extend(["\nEventos: No hay eventos programados para las fechas de tu viaje."])
            else:
                itinerario_texto_lista.extend(["\nEventos:"])
                itinerario_texto_lista.extend(eventos_sel["Descripcion"].tolist())

            itinerario_txt = "\n".join(itinerario_texto_lista)


            print('itinerary end... ',round(time.time()-start,2))
            print('---')
            # Almacena el itinerario generado en el estado de sesión
            st.session_state.texto_itinerario = "\n".join(itinerario_texto_lista)
            st.session_state.llm_reformatted_itinerario = None # Resetear el LLM reformateado
            st.session_state.show_download_original = True # Mostrar botón de descarga original
            st.session_state.show_download_llm = False # Ocultar botón de descarga LLM hasta que se genere
            st.session_state.execute_llm_model = False

            st.success("¡Itinerario base generado! Ahora puedes verlo y reformatearlo con IA.")
            st.rerun() # Fuerza una re-ejecución para mostrar el itinerario y los botones
            
#####################################################################################################################################
################################################## VISUALIZACIÓN Y OPCIONES DEL ITINERARIO ################################
#####################################################################################################################################

if st.session_state.texto_itinerario:
    # st.subheader("Itinerario Base Generado:")
    # st.text_area("Aquí está el itinerario generado con los parámetros seleccionados.", st.session_state.texto_itinerario, height=300, key="base_itinerary_display")

    # col1, col2 = st.columns(2) # Dos columnas para el botón de LLM y el de descarga original

    if st.session_state.show_download_original:
        st.subheader("🗺️ Itinerario Base:")
        nombre_archivo_original = st.text_input("Nombre para el archivo original (sin .txt)", value="itinerario_original", key="fname_orig")
        st.download_button(
            label="📥 Descargar Itinerario Original",
            data=st.session_state.texto_itinerario,
            file_name=f"{nombre_archivo_original}.txt" if nombre_archivo_original else "itinerario_original.txt",
            mime="text/plain",
            key="download_orig_btn",
            help="Descarga el itinerario tal como fue generado inicialmente."
        )


 
    if st.sidebar.button("✨ Reformatear con LLM"):
        if not st.session_state.texto_itinerario:
            st.error("Primero genera un itinerario base para poder reformatearlo.")
        elif not pipe:
            st.error("Añade tu nombre de usuario o descarga el modelo Llama-3.2-3B-Instruct.")
        else:
                st.session_state.execute_llm_model=True
                # Importaciones de transformers se hacen aquí para que solo se carguen cuando se necesite

    # with col1:
    #     if st.button("✨ Reformatear con LLM"):
    #         if not st.session_state.texto_itinerario:
    #             st.error("Primero genera un itinerario base para poder reformatearlo.")
    #         elif not pipe:
    #             st.error("Añade tu nombre de usuario o descarga el modelo Llama-3.2-3B-Instruct.")
    #         else:
    #                 st.session_state.execute_llm_model=True
    #                 # Importaciones de transformers se hacen aquí para que solo se carguen cuando se necesite
    # with col2:
    #     if st.session_state.show_download_original:
    #         nombre_archivo_original = st.text_input("Nombre para el archivo original (sin .txt)", value="itinerario_original", key="fname_orig")
    #         st.download_button(
    #             label="📥 Descargar Itinerario Original",
    #             data=st.session_state.texto_itinerario,
    #             file_name=f"{nombre_archivo_original}.txt" if nombre_archivo_original else "itinerario_original.txt",
    #             mime="text/plain",
    #             key="download_orig_btn",
    #             help="Descarga el itinerario tal como fue generado inicialmente."
    #         )
#####################################################################################################################################
################################################## GENERAR LLM ################################
#####################################################################################################################################
if st.session_state.executed_response:
    #st.session_state.texto_itinerario = "\n".join(itinerario_texto_lista)
    st.session_state.llm_reformatted_itinerario = None # Resetear el LLM reformateado
    st.session_state.show_download_original = True # Mostrar botón de descarga original
    st.session_state.show_download_llm = False # Ocultar botón de descarga LLM hasta que se genere
    st.session_state.execute_llm_model = False
    st.session_state.executed_response=False

# Sección para mostrar el itinerario reformulado por el LLM
if st.session_state.execute_llm_model:


        #st.stop()

    st.subheader("✨ Itinerario Reformulado y Mejorado por IA:")
    # Usamos st.markdown para renderizar el texto con formato (negritas, encabezados)
    prompt_List=st.session_state.texto_itinerario.split('\n')
    system1=""" Eres un recomendador de planes turisticos agradable y con energía. Cuando te digan un elemento quiero que lo reformules haciendolo algo más breve pero que sea atractivo
    para un nuevo usario que lo lea. Cita siempre el nombre del elemento en negrita. Si puedes introducelo todo en dos lineas. No inventes ."""
    system_intro=""" Eres un recomendador de planes turisticos agradable y con energía. Cuando te hablen genera una introducción o conlusión en una linea. Esta debe estar orientada en Asturias"""
    # principal=['- Hoteles:'] # ,'Desayuno:','Comida:','Cena:','Lugares turísticos'
    # 🔹 Detectamos si existe la sección de "Eventos:"
    if "Eventos:" in prompt_List:
        idx_eventos = prompt_List.index("Eventos:")
        seccion_reformatear = prompt_List[:idx_eventos]
        seccion_eventos = prompt_List[idx_eventos:]
    else:
        seccion_reformatear = prompt_List
        seccion_eventos = []

    with st.spinner("Preparando el modelo de IA... Esto puede tardar un poco la primera vez que se carga. Por favor, sé paciente."):
        respuesta =LLM_output(pipe,system_intro,"Introducción",200)
        st.session_state.llm_reformatted_itinerario=""" """
        st.session_state.llm_reformatted_itinerario+=respuesta+"\n"
        st.write(respuesta)
        st.write("")
        for mini_prompt in seccion_reformatear:  
            if 'RUTA' in mini_prompt or "Precio diario" in mini_prompt or "Precio total:" in mini_prompt  or mini_prompt=='':# or mini_prompt in principal
                # st.write()
                if 'RUTA' in mini_prompt or "Precio diario" in mini_prompt or "Precio total:" in mini_prompt : # or mini_prompt in principal
                    st.session_state.llm_reformatted_itinerario+=mini_prompt+"\n"
                    st.write(mini_prompt)
                else:
                    st.session_state.llm_reformatted_itinerario+="\n"

            else:
                respuesta =LLM_output(pipe,system1,mini_prompt,200)
                st.write(respuesta)
                st.session_state.llm_reformatted_itinerario+=respuesta+"\n"
        
        if seccion_eventos:
            st.write("")
            for linea in seccion_eventos:
                st.write(linea)
                st.session_state.llm_reformatted_itinerario += linea + "\n"
            
        st.write("")
        respuesta =LLM_output(pipe,system_intro,"Conclusión",200)   
        st.write(respuesta)
        st.session_state.llm_reformatted_itinerario+=respuesta
        print(st.session_state.llm_reformatted_itinerario)

    st.session_state.show_download_llm=True

    if st.session_state.show_download_llm:
        nombre_archivo_llm = st.text_input("Nombre para el archivo LLM (sin .txt)", value="itinerario_asturias_LLM", key="fname_llm")
        
        # Limpiar el texto del LLM para la descarga (eliminar markdown)
        #texto_llm_para_descarga = re.sub(r'##\s*', '', st.session_state.llm_reformatted_itinerario) # Elimina encabezados Markdown
        texto_llm_para_descarga = re.sub(r'\*\*', '', st.session_state.llm_reformatted_itinerario) # Elimina negritas Markdown

        st.download_button(
            label="📥 Descargar Itinerario LLM",
            data=texto_llm_para_descarga,
            file_name=f"{nombre_archivo_llm}.txt" if nombre_archivo_llm else "itinerario_asturias_LLM.txt",
            mime="text/plain",
            key="download_llm_btn",
            help="Descarga la versión del itinerario mejorada por la IA."
        )
        st.session_state.executed_response=True

# Mensaje de bienvenida inicial
elif not st.session_state.texto_itinerario and not st.session_state.llm_reformatted_itinerario:
    st.info("👋 ¡Bienvenido! Configura tu viaje en la barra lateral izquierda y pulsa 'Generar Itinerario' para empezar tu aventura por Asturias.")