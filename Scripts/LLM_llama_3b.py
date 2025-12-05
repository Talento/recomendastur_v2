import os
import re
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig, pipeline
# Configuración de la caché de Hugging Face

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

# Este 'prompt' se procesará línea por línea para generar recomendaciones.
prompt="""

RUTA 1 - Zona Oviedo, Día 1
Alojamiento: Hotel Castro Real - Precio de 48.0 - Valoración de 8.4 - Coordenadas de (43.373905702361455, -5.835161805152893)
Desayuno: Cafe-Bar Jota L 28 - Precio de 10.0 - Valoración de 78.0 - Coordenadas de (43.3634867, -5.866514) - (3.3 km, 48 mins, Andando)
Lugar turístico: Vía Verde de Fuso - Precio de nan - Valoración de nan - Coordenadas de (43.33544099547839, -5.886645181910597) - (4.8 km, 1 hour 10 mins, Andando)
Comida: Bar Pulpería Cares - Precio de 16.0 - Valoración de 90.0 - Coordenadas de (43.3684328, -5.8637326) - (5.5 km, 1 hour 21 mins, Andando)
Lugar turístico: Parque de invierno - Precio de nan - Valoración de nan - Coordenadas de (43.351097, -5.848528) - (2.8 km, 41 mins, Andando)
Cena: La Finca Sidrería Agrobar - Precio de 16.0 - Valoración de 88.0 - Coordenadas de (43.3641577, -5.8441813) - (1.9 km, 27 mins, Andando)

RUTA 2 - Zona Oviedo, Día 1
Alojamiento: Astures - Precio de 62.0 - Valoración de 8.4 - Coordenadas de (43.36379487567506, -5.839375555515289)
Desayuno: Restaurante Cafetería Mar del Plata - Precio de 10.0 - Valoración de 72.0 - Coordenadas de (43.361574, -5.856246) - (1.6 km, 24 mins, Andando)
Lugar turístico: Llagar Herminio - Precio de nan - Valoración de nan - Coordenadas de (43.37716, -5.804) - (4.9 km, 1 hour 6 mins, Andando)
Comida: UMAMI - Precio de 17.0 - Valoración de 92.0 - Coordenadas de (43.3637458, -5.8658757) - (5.7 km, 1 hour 23 mins, Andando)
Lugar turístico: Cofradía de Doña Gontrodo - Precio de nan - Valoración de nan - Coordenadas de (43.361263, -5.854911) - (1.0 km, 15 mins, Andando)
Cena: Restaurante - Pizzería Salvatore - Precio de 17.0 - Valoración de 88.0 - Coordenadas de (43.3629937, -5.861021) - (0.6 km, 9 mins, Andando)

RUTA 1 - Zona Oviedo, Día 2
Alojamiento: Ibis Budget Oviedo - Precio de 43.0 - Valoración de 7.7 - Coordenadas de (43.37365809449615, -5.850241184234619)
Desayuno: Café Restaurante Que Idea - Precio de 10.0 - Valoración de 80.0 - Coordenadas de (43.357196, -5.8714208) - (2.8 km, 42 mins, Andando)
Lugar turístico: Cámara Santa - Precio de nan - Valoración de nan - Coordenadas de (43.362488, -5.843598) - (2.6 km, 35 mins, Andando)
Comida: Bar Pulpería Cares - Precio de 16.0 - Valoración de 90.0 - Coordenadas de (43.3684328, -5.8637326) - (2.0 km, 29 mins, Andando)
Lugar turístico: Catedral de El Salvador - Precio de nan - Valoración de nan - Coordenadas de (43.362439, -5.843627) - (2.0 km, 28 mins, Andando)
Cena: Sidrería Pichote - Precio de 18.0 - Valoración de 88.0 - Coordenadas de (43.3681001, -5.8680138) - (2.4 km, 34 mins, Andando)

RUTA 2 - Zona Oviedo, Día 2
Alojamiento: Oviedo centro - Precio de 84.0 - Valoración de 9.3 - Coordenadas de (43.3664709, -5.852597)
Desayuno: Cafetería El Carmen - Precio de 10.0 - Valoración de 70.0 - Coordenadas de (43.3839258, -5.8238157) - (3.5 km, 46 mins, Andando)
Lugar turístico: Iglesia de San Julián de los Prados - Precio de nan - Valoración de nan - Coordenadas de (43.367712, -5.837325) - (2.5 km, 36 mins, Andando)
Comida: El Llagarín de Granda - Precio de 16.0 - Valoración de 88.0 - Coordenadas de (43.3833062, -5.7759329) - (5.8 km, 1 hour 20 mins, Andando)
Lugar turístico: Monasterio de San Vicente - Precio de nan - Valoración de nan - Coordenadas de (43.363051, -5.842179) - (6.0 km, 1 hour 26 mins, Andando)
Cena: El Rincón de Adi - Precio de 17.0 - Valoración de 88.0 - Coordenadas de (43.3658182, -5.8452131) - (0.5 km, 6 mins, Andando)"""

prompt_List=prompt.split('\n')
system1=""" Eres un recomendador de planes turisticos agradable y con energía. Cuando te digan un elemento quiero que lo reformules haciendolo algo más breve pero que sea atractivo
para un nuevo usario que lo lea. Cita siempre el nombre del elemento en negrita. Si puedes introducelo todo en dos lineas. No inventes ."""
system_intro=""" Eres un recomendador de planes turisticos agradable y con energía. Cuando te hablen genera una introducción o conlusión en una linea. Esta debe estar orientada en Asturias"""
principal=['- Hoteles:','Desayuno:','Comida:','Cena:','Lugares turísticos:']

LLM_output(system_intro,"Introducción",200)
print()
for mini_prompt in prompt_List[:]:
    if 'Día' in mini_prompt or mini_prompt in principal or mini_prompt=='':
        print(mini_prompt)
    else:
        LLM_output(system1,mini_prompt,200)
print()
LLM_output(system_intro,"Conclusión",200)