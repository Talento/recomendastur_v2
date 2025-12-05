import pandas as pd
import warnings
import sys
import os
import re
import torch

from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig, pipeline # Configuración de la caché de Hugging Face
from Functions_itinerary import solicitar_fecha, obtener_coordenadas, seleccionar_datos_viaje, extraer_primera_frase, generar_texto_ruta, obtener_tiempo_y_distancia, anotar_tramo

warnings.filterwarnings("ignore")  # Suprime advertencias
sys.dont_write_bytecode = True     # Evita creación de archivos .pyc

# ============================== CARGA DE DATOS =============================
# Carga los datasets de alojamientos, restaurantes y puntos turísticos
alojamientos = pd.read_csv("./Definitive_data/accommodation_processed.csv")
restaurantes = pd.read_csv("./Definitive_data/restaurant_processed.csv")
turistico = pd.read_csv("./Definitive_data/tourist_processed.csv")
cities = pd.read_csv("./Definitive_data/cities_coords.csv")

# ============================== INPUTS DE USUARIO ==========================

# Solicita fecha de inicio y fin del viaje
while True:
        fecha_inicio_str = input("Introduce la fecha de inicio del viaje (dd-mm-yyyy) o deja en blanco para usar hoy: ")
        if not fecha_inicio_str:
            fecha_inicio = datetime.now()
            print(f"Usando fecha de inicio por defecto: {fecha_inicio.strftime('%d-%m-%Y')}")
            break
        try:
            fecha_inicio = datetime.strptime(fecha_inicio_str, "%d-%m-%Y")
            break
        except ValueError:
            print("Formato de fecha incorrecto. Por favor, usa dd-mm-yyyy.")
while True:
        fecha_fin_str = input(f"Introduce la fecha de fin del viaje (dd-mm-yyyy, posterior a {fecha_inicio.strftime('%d-%m-%Y')}): ")
        if not fecha_fin_str: # Opción para itinerario de 1 día por defecto si se deja en blanco
            fecha_fin = fecha_inicio 
            print(f"Usando fecha de fin por defecto (viaje de 1 día): {fecha_fin.strftime('%d-%m-%Y')}")
            break
        try:
            fecha_fin = datetime.strptime(fecha_fin_str, "%d-%m-%Y")
            if fecha_fin >= fecha_inicio:
                break
            else:
                print("La fecha de fin debe ser igual o posterior a la fecha de inicio.")
        except ValueError:
            print("Formato de fecha incorrecto. Por favor, usa dd-mm-yyyy.")

dias = (fecha_fin - fecha_inicio).days + 1

# Solicita tipo de viaje y recoge número de personas y si viajan niños
while True:
    tipo_viaje = input('Introduce el tipo de viaje ["Solo", "Pareja", "Familia", "Amigos"](def: Pareja): ').strip().capitalize()
    if not tipo_viaje: tipo_viaje = "Pareja"  # Valor por defecto si se deja en blanco
    if tipo_viaje in ["Solo", "Pareja", "Familia", "Amigos"]:
        if tipo_viaje == "Solo":
            n_personas = 1 
            ninos = "N"
        elif tipo_viaje == "Pareja":
            n_personas = 2
            ninos = "N"
        else: # Para "Familia" o "Amigos"
            while True:
                try:
                    n_personas_str = input(f"Ingrese el número de personas para '{tipo_viaje}' (def: 3): ").strip()
                    if not n_personas_str:
                        n_personas = 3  # o el valor por defecto que quieras
                        break
                    else:
                        n_personas = int(n_personas_str)
                        if n_personas <= 0:
                            print("Error: El número de personas debe ser mayor que 0.")
                            continue
                        break  # ← ¡este sí se alcanza si todo va bien!
                except ValueError:
                    print("Error: Debe ingresar un número válido.")

            if tipo_viaje == "Familia": # Solo preguntar por niños si es Familia
                    while True:
                        ninos = input("¿Viajan niños? (S/N) (def: N): ").strip().upper()
                        if not ninos: ninos = "N"
                        if ninos in ["S", "N"]:
                            break
                        else:
                            print("Error: Debe ingresar 'S' para sí o 'N' para no.")
            else: # Amigos
                ninos = "N"
        break
    else:
        print(f"'{tipo_viaje}' no es una opción válida.")

salir_zona_coche = False
while not salir_zona_coche:
    coche = input("¿Dispones de coche? (S/N) (def: S): ").strip().upper()
    if not coche: coche = "S"
    zonas = [] # Lista de zonas del viaje
    dias_zona = [] # Número de días a pasar en cada zona
    
    if coche == "S":
        while True:
            try:
                radio_str = input("¿A qué distancia máxima te quieres desplazar desde tu zona de alojamiento? (km, def: 30): ")
                if not radio_str: radio = 30.0
                else: radio = float(radio_str)
                
                if radio <= 5: # Ajustado a >5 km.
                    print("Error: El radio tiene que ser mayor de 5 km.")
                    continue
                break
            except ValueError:
                print("Error: Debe ingresar un número válido para el radio.")
        
        while True: # Bucle para tipo de alojamiento (fijo o múltiple)
            zona_hotel_fijo_str = input("¿Quieres alojarte en la misma zona todos los días ('S') o en diferentes zonas ('N')? (S/N, def: S): ").strip().upper()
            if not zona_hotel_fijo_str: zona_hotel_fijo_str = "S"

            if zona_hotel_fijo_str == "S":
                while True:
                    zona_unica = input("Introduce la zona principal del viaje (ej: Gijon, Oviedo, Aviles, Llanes, Cangas de Onis... def: Gijon): ").strip()
                    if not zona_unica: zona_unica = "Gijon"
                    if obtener_coordenadas(zona_unica, cities): # Usa la función (placeholder o real)
                        zonas = [zona_unica]
                        dias_zona = [dias] # Todos los días en esta única zona
                        salir_zona_coche = True
                        break # Sale del bucle de zona_unica
                    else:
                        print(f"Error: La zona '{zona_unica}' no se pudo validar. Intenta con otra conocida de Asturias.")
                if salir_zona_coche: break # Sale del bucle de tipo de alojamiento
            
            elif zona_hotel_fijo_str == "N":
                dias_asignados = 0
                while dias_asignados < dias:
                    print(f"Días totales del viaje: {dias}. Días ya asignados a zonas: {dias_asignados}. Días restantes por asignar: {dias - dias_asignados}")
                    while True:
                        zona_actual = input(f"Introduce la siguiente zona del viaje: ").strip()
                        if obtener_coordenadas(zona_actual):
                            break
                        else:
                            print(f"Error: La zona '{zona_actual}' no se pudo validar. Intenta con otra conocida de Asturias.")
                    
                    while True:
                        try:
                            n_dias_en_zona_actual_str = input(f"¿Cuántos días quieres pasar en {zona_actual}? (máx {dias - dias_asignados} días): ")
                            n_dias_en_zona_actual = int(n_dias_en_zona_actual_str)
                            if n_dias_en_zona_actual <= 0:
                                print("Error: El número de días debe ser mayor a 0.")
                            elif (dias_asignados + n_dias_en_zona_actual) > dias:
                                print(f"Error: Superas el total de días del viaje. Solo te quedan {dias - dias_asignados} días disponibles.")
                            else:
                                break # Días válidos para esta zona
                        except ValueError:
                            print("Error: Debe ingresar un número entero válido para los días.")
                    
                    zonas.append(zona_actual)
                    dias_zona.append(n_dias_en_zona_actual)
                    dias_asignados += n_dias_en_zona_actual
                
                if dias_asignados == dias:
                    print("Todas las zonas y días asignados correctamente.")
                    salir_zona_coche = True
                    break # Sale del bucle de tipo de alojamiento (S/N)
                else:
                    # Esto no debería ocurrir si la lógica anterior es correcta, pero por si acaso.
                    print(f"Error en la asignación de días. Total asignado: {dias_asignados}, Días de viaje: {dias}. Reiniciando selección de zonas múltiples.")
                    zonas = [] # Reset
                    dias_zona = [] # Reset
                    # No salir_zona_coche, se vuelve a preguntar S/N
            else:
                print("Opción no válida. Introduce 'S' para zona fija o 'N' para múltiples zonas.")
        # Fin bucle tipo de alojamiento (S/N)

    elif coche == "N":
        radio = 5.0 # Radio fijo y pequeño si no hay coche
        while True:
            zona_unica_sin_coche = input("Introduce la zona del viaje, ej: Gijon, Oviedo... def: Gijon): ").strip()
            if not zona_unica_sin_coche: zona_unica_sin_coche = "Gijon"
            if obtener_coordenadas(zona_unica_sin_coche):
                zonas = [zona_unica_sin_coche]
                dias_zona = [dias]
                salir_zona_coche = True
                break # Sale del bucle de zona_unica_sin_coche
            else:
                print(f"Error: La zona '{zona_unica_sin_coche}' no se pudo validar.")
    else:
        print("Opción no válida para coche. Introduce S o N.")

# ========================= GENERACIÓN DEL ITINERARIO =======================

# Selecciona lugares relevantes en base a preferencias y disponibilidad
alojamientos_seleccion, turistico_seleccion, cafeterias_seleccion, restaurantes_seleccion = seleccionar_datos_viaje(
    alojamientos, turistico, restaurantes, cities, zonas, dias_zona, n_personas, radio, ninos
)

# Extrae frases resumidas de las descripciones
alojamientos_seleccion['Primera_Frase'] = alojamientos_seleccion['Descripción'].apply(extraer_primera_frase)
turistico_seleccion['Primera_Frase'] = turistico_seleccion['Descripción'].apply(extraer_primera_frase)

# Variables de control
indice_aloj = 0
indice_turi = 0
rutas_conjuntas = []
itinerario = []
dia_global = 1

# Se generan dos rutas por día, una para cada alternativa (ruta 1 y 2)
for zona in range(len(zonas)):
    dias = dias_zona[zona]
    aloj_zona = alojamientos_seleccion.iloc[indice_aloj : indice_aloj + 2*dias].reset_index(drop=True)
    cafe_zona = cafeterias_seleccion.iloc[indice_aloj : indice_aloj + 2*dias].reset_index(drop=True)
    rest_zona = restaurantes_seleccion.iloc[indice_turi : indice_turi + 4*dias].reset_index(drop=True)
    turi_zona = turistico_seleccion.iloc[indice_turi : indice_turi + 4*dias].reset_index(drop=True)

    for i in range(dias):
        # Ruta 1
        ruta1 = pd.concat([
            aloj_zona.iloc[2*i:2*i+1], cafe_zona.iloc[2*i:2*i+1],
            turi_zona.iloc[4*i:4*i+1], rest_zona.iloc[4*i:4*i+1],
            turi_zona.iloc[4*i+1:4*i+2], rest_zona.iloc[4*i+1:4*i+2]
        ], ignore_index=True)
        ruta1['Zona'] = zonas[zona]
        ruta1['Día'] = dia_global
        ruta1['Ruta'] = 1
        rutas_conjuntas.append(ruta1)
        itinerario.extend(generar_texto_ruta(ruta1, zonas[zona], dia_global, 1, coche))

        # Ruta 2
        ruta2 = pd.concat([
            aloj_zona.iloc[2*i+1:2*i+2], cafe_zona.iloc[2*i+1:2*i+2],
            turi_zona.iloc[4*i+2:4*i+3], rest_zona.iloc[4*i+2:4*i+3],
            turi_zona.iloc[4*i+3:4*i+4], rest_zona.iloc[4*i+3:4*i+4]
        ], ignore_index=True)
        ruta2['Zona'] = zonas[zona]
        ruta2['Día'] = dia_global
        ruta2['Ruta'] = 2
        rutas_conjuntas.append(ruta2)
        itinerario.extend(generar_texto_ruta(ruta2, zonas[zona], dia_global, 2, coche))
        dia_global += 1

    indice_aloj += 2 * dias
    indice_turi += 4 * dias

# Genera texto completo del itinerario
texto = "\n".join(itinerario)

# ======================== SALIDA DEL ITINERARIO ============================

# Permite guardar el itinerario como archivo de texto
guardar = input("¿Quieres guardar el itinerario generado en txt? (S/N): ").strip().upper()
if guardar == "S":
    nombre_archivo = input("¿Qué nombre le quieres dar al archivo ? ").strip()
    file_path = f"{nombre_archivo}.txt"  # << Guarda aquí la ruta del archivo para usarla después
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(texto)
    print(f"El itinerario ha sido guardado como {file_path}")
else:       
    print("El itinerario no se ha guardado.")
    file_path = None  # << Para controlar luego si se puede cargar el archivo

# ============================== LLAMA LLM ====================================

# Rutas del modelo
local_model_path="C:/Users/carlos.redondo/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"  # Ruta local del modelo en caché
remote_model_path="meta-llama/Llama-3.2-3B-Instruct"
auth_token="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"  # Token de autenticación para modelos privados en este caso de Llama-2-13b-chat-hf

# Verificar si hay GPU disponible
if torch.cuda.is_available():
    print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")  # Muestra la GPU detectada
else:
    print("⚠️ No se está utilizando GPU. Verifica tu instalación de CUDA y PyTorch.") # Comprueba si hay GPU

# Verificar si el modelo ya está en caché
if os.path.exists(local_model_path):
    print("✅ Modelo encontrado en caché. Cargando desde disco...")
    model_path=local_model_path  # Usa el modelo almacenado localmente
else:
    print("⚠️ Modelo no encontrado en caché. Descargando...")
    model_path=remote_model_path  # Si no está en caché, usa la ruta remota que le hemos proporcionado

# Verificar si el tokenizador está en caché
tokenizer_path = local_model_path if os.path.exists(local_model_path) else remote_model_path

# Cargar el tokenizador
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path,
                                           trust_remote_code=True,
                                            token=auth_token)

# Ajustes del tokenizador
tokenizer.pad_token = tokenizer.eos_token  # Usa el token de fin de secuencia (EOS) como padding
tokenizer.padding_side = "right"  # Padding a la derecha para modelos autoregresivos, los modelos como llama siempre son right

# Configuración de bitsandbytes para carga optimizada
bnb_config = BitsAndBytesConfig(load_in_4bit=True,  # Reduce el tamaño del modelo a 4 bits
                                bnb_4bit_quant_type="nf4",  # Usa Normal Float 4 para más precisión
                                bnb_4bit_compute_dtype=torch.float16,  # Realiza cálculos en FP16
                                bnb_4bit_use_double_quant=False)  # No usa doble cuantización

# Cargar modelo con distribución automática entre GPU y CPU
model = AutoModelForCausalLM.from_pretrained(model_path, 
                                             token=auth_token,
                                             rope_scaling={"type": "dynamic", "factor": 2},  # Expande contexto RoPE, multiplica por dos el numero de tokens
                                             quantization_config=bnb_config,  # Aplica configuración de cuantización
                                             device_map={"": 0})  # Distribuye el modelo en la GPU disponible

# Configurar el streamer de texto para salida en tiempo real
streamer = TextStreamer(tokenizer, # Utiliza el tokenizador para decodificar los tokens generados en texto legible.
                        skip_prompt=True,     # Omite la impresión del prompt original en la salida generada.
                        skip_special_tokens=True  # Elimina los tokens especiales como <s>, </s>, [PAD], etc.
)

# Función para generar texto con el modelo

pipe = pipeline(
    "text-generation",  # Tipo de tarea: generación de texto.
    model=model,        # Instancia del modelo cargado.
    tokenizer=tokenizer, # Instancia del tokenizador cargado.
    torch_dtype=torch.bfloat16, # Tipo de dato para los cálculos internos del pipeline.
    device_map=0,       # Asegura que el pipeline también use la GPU 0 para la generación.
    streamer=streamer,  # Conecta el streamer para la salida en tiempo real.
)

# Definición de la función `LLM_output` para interactuar con el modelo de lenguaje.
def LLM_output(system, prompt, max_new_tokens):
    messages = [
        {"role": "system", "content": system}, # system, como quiero que actue
        {"role": "user", "content": prompt}, # promt, lo que le pido
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        do_sample=True
    )


# Procesamiento de Rutas desde Archivo
if file_path:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        print(f"✅ Contenido cargado exitosamente desde '{file_path}'")
    except FileNotFoundError:
        print(f"❌ Error: El archivo '{file_path}' no se encontró. Asegúrate de que la ruta sea correcta.")
        prompt_content = ""
    except Exception as e:
        print(f"❌ Ocurrió un error al leer el archivo '{file_path}': {e}")
        prompt_content = ""
else:
    print("❌ No se ha definido un archivo para cargar.")
    prompt_content = ""

# Dividir el contenido del archivo en una lista de líneas individuales
prompt_List = prompt_content.split('\n')

# Definición de las instrucciones para el "system" (rol del chatbot).
system1 = """ Eres un recomendador de planes turisticos agradable y con energía. Cuando te digan un elemento quiero que lo reformules haciendolo algo más breve pero que sea atractivo
para un nuevo usario que lo lea. Cita siempre el nombre del elemento en negrita. Si puedes introducelo todo en dos lineas. No inventes ."""
system_intro = """ Eres un recomendador de planes turisticos agradable y con energía. Cuando te hablen genera una introducción o conlusión en una linea. Esta debe estar orientada en Asturias"""

# Lista de palabras clave para identificar los encabezados principales dentro del prompt.
principal = ['Alojamiento:','Desayuno:','Lugar turístico:','Comida:','Lugares turísticos:','Cena:']

LLM_output(system_intro,"Introducción",200)
print()
for mini_prompt in prompt_List[:]:
    if 'Día' in mini_prompt or mini_prompt in principal or mini_prompt=='' or "Precio diario" in mini_prompt or "Precio aproximado" in mini_prompt:
        print(mini_prompt)
    else:
        LLM_output(system1,mini_prompt,200)
print()
LLM_output(system_intro,"Conclusión",200)