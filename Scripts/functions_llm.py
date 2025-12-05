'''
import os
import torch
import streamlit as st

from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig, pipeline

from dotenv import load_dotenv
# Cargar las variables del archivo .env
load_dotenv()
USER_NAME = os.getenv("USER_NAME")

@st.cache_resource
def load_llm(remote=False):
    print(USER_NAME)
    local_model_path = f"C:/Users/{USER_NAME}/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
    auth_token = "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" 
    remote_model_path="meta-llama/Llama-3.2-3B-Instruct"

    # Verificar si hay GPU disponible
    if torch.cuda.is_available():
        print(f"✅ GPU detectada: {torch.cuda.get_device_name(0)}")  # Muestra la GPU detectada
    else:
        print("⚠️ No se está utilizando GPU. Verifica tu instalación de CUDA y PyTorch.") # Comprueba si hay GPU

    # Verificar si el modelo ya está en caché
    if os.path.exists(local_model_path):
        print("✅ Modelo encontrado en caché. Cargando desde disco...")
        model_path=local_model_path
    else:
        if not remote:
            print("⚠️ Modelo no encontrado en disco, verifica la ruta o descarga el modelo...")
            return None
        else:
            print("⚠️ Descargando modelo...")
            model_path=remote_model_path

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
    # Cargar modelo
    model = AutoModelForCausalLM.from_pretrained(model_path, 
                                                 token=auth_token,
                                                 rope_scaling={"type": "dynamic", "factor": 2},  # Expande contexto RoPE, multiplica por dos el numero de tokens
                                                 quantization_config=bnb_config,  # Aplica configuración de cuantización
                                                 device_map={"": 0},  #gpu cuda
                                                temperature=0.9
    )

    streamer = TextStreamer(tokenizer, # para decodificar los tokens generados en texto legible.
                        skip_prompt=True,     # Omite la impresión del prompt original en la salida generada.
                        skip_special_tokens=True  # Elimina los tokens especiales como <s>, </s>, [PAD], etc.
    )

    pipe = pipeline(
    "text-generation",
    model=model,#modelo
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map=0,#usar gpu cuda
    #streamer=streamer,
    )

    return pipe


def LLM_output(pipe,system,prompt,max_new_tokens):
    messages = [
        {"role": "system", "content":system },#system, como quiero que actue
        {"role": "user", "content":prompt },#promt, lo que le pido
    ]
    outputs = pipe(
        messages,
        max_new_tokens=max_new_tokens,
    )
    #print('-----------------------',outputs[0].keys())
    return outputs[0]['generated_text'][2]['content']

    '''


import os
import streamlit as st
import ollama # <--- La única librería que necesitas ahora para la IA

from dotenv import load_dotenv
# Cargar las variables del archivo .env
load_dotenv()
USER_NAME = os.getenv("USER_NAME")

@st.cache_resource
def load_llm(remote=False):
    print(USER_NAME)
    
    # En Ollama no necesitamos rutas de archivos complicadas ni tokens de HF.
    # Solo necesitamos saber el nombre del modelo que descargaste en la terminal.
    # Asegúrate de haber hecho 'ollama run llama3.2' antes.
    model_name = "llama3.2" 

    try:
        # Hacemos una pequeña prueba de conexión (opcional, pero recomendada)
        ollama.show(model_name)
        print(f"✅ Conectado exitosamente con Ollama. Modelo: {model_name}")
    except Exception as e:
        print(f"⚠️ Error: No se detecta el modelo '{model_name}'. Ejecuta 'ollama run {model_name}' en tu terminal.")
        return None

    # En lugar de devolver un objeto 'pipeline' complejo y pesado, 
    # devolvemos el nombre del modelo. Esto actuará como tu 'pipe'.
    return model_name


def LLM_output(pipe, system, prompt, max_new_tokens):
    # 'pipe' ahora es el string "llama3.2" que nos dio la función de arriba.
    
    messages = [
        {"role": "system", "content": system}, # system, como quiero que actue
        {"role": "user", "content": prompt},   # prompt, lo que le pido
    ]

    # Llamada a la API de Ollama
    # Mapeamos 'max_new_tokens' a 'num_predict' que es como lo llama Ollama
    response = ollama.chat(
        model=pipe, 
        messages=messages,
        options={
            "num_predict": max_new_tokens, 
            "temperature": 0.9 # Mantenemos la temperatura que tenías antes
        }
    )

    # Devolvemos directamente el contenido limpio.
    # Tu código antiguo tenía que navegar por listas complejas ([0]['generated_text']...),
    # Ollama nos lo da más fácil:
    return response['message']['content']