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


# Procesamiento de Rutas desde Archivo
# Nombre del archivo de texto que contiene tus rutas
file_path = "test_2.txt" # Asegúrate de que este archivo esté en el mismo directorio que tu script, o especifica la ruta completa.

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        prompt_content = f.read()
    print(f"✅ Contenido cargado exitosamente desde '{file_path}'")
except FileNotFoundError:
    print(f"❌ Error: El archivo '{file_path}' no se encontró. Asegúrate de que la ruta sea correcta.")
    prompt_content = "" # O manejar el error de otra manera, quizás salir del script.
except Exception as e:
    print(f"❌ Ocurrió un error al leer el archivo '{file_path}': {e}")
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
    if 'Día' in mini_prompt or mini_prompt in principal or mini_prompt=='':
        print(mini_prompt)
    else:
        LLM_output(system1,mini_prompt,200)
print()
LLM_output(system_intro,"Conclusión",200)