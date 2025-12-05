from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time
import warnings
from functions_web_scraping import data_extractor_selenium
warnings.filterwarnings("ignore")
######################################50#############################################################################
################################            DATOS            ######################################################
###################################################################################################################
df = pd.read_csv('Data/Data_used/restaurant_urls.csv') 
option = webdriver.ChromeOptions()
#option.add_argument("--incognito")
option.add_argument('--headless=new')#para que no se abra el navegador
###################################################################################################################
################################            INPUTS            #####################################################
###################################################################################################################
def inputs_():
    enlace_ini=int(input('Indice del enlace de inicio?   '))
    correct=input(f'Estas seguro que quieres empezar por el enlace {df.urls.iloc[enlace_ini]}\ny/n   ')
    enlace_lim=int(input('Cuantos enlaces quieres procesar?   '))
    if correct=='y':
        return enlace_ini,enlace_lim
    else:
        return inputs_()
ini,lim=inputs_()
###################################################################################################################
################################            MAIN            ######################################################
###################################################################################################################
# Lista para almacenar los datos de cada restaurante (cada restaurante es un elemento)
resultados = []
start_0=time.time()
# Inicializamos el driver
#options = Options()
# options.add_argument('--headless=new')  # Ejecutar el navegador en modo headless
driver = webdriver.Chrome(options=option)

id=ini
# Seleccionamos el primer alojamiento y cargamos la página correspondiente
restaurante = df.iloc[id]
start=time.time()
driver.get(restaurante["urls"])
time.sleep(2)

# Aceptamos las cookies
cookies= driver.find_element(By.XPATH,'//button[@aria-label="Consent"]')
cookies.click()
time.sleep(1)
# Extraemos los resultados para el primer restaurante y los añadimos a la lista de datos
titulo, lat, long, loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig,driver=data_extractor_selenium(driver,restaurante["urls"],id,start)
resultados.append({
        'Nombre': titulo, 'Localización': localidad, 'Latitud': lat, 'Longitud': long,
        'Dirección': loc, "Teléfono": telefono, "Precio": precios, 'Horario': horario,
        "Tipos de cocina": tipo_cocina, "Valoración": valora, "Número de votos": votos, 
        "Características": caracteristicas, "Descripción": descripcion, "url":restaurante["urls"],
        'url_web':enlace_web,'url_ig':enlace_ig
})

# Sin cerrar el navegador, se accede a los demás restaurantes, repitiendo el proceso de obtención y guardado de datos.
for id in range(ini+1, ini+lim):
    if id>=len(df):
        print("Limite de URLS")
        break
    start=time.time()
    restaurante = df.iloc[id] # Seleccionar el restaurante.
    driver.execute_script(f'window.open("{restaurante["urls"]}", "_blank");') # Abrir la nueva url
    time.sleep(1)
    driver.close() # Cerrar la url anterior
    time.sleep(1)
    driver.switch_to.window(driver.window_handles[0]) # Moverse a la nueva página
    
    time.sleep(1)

    # Extraer los resultados para cada alojamiento y añadirlos a la lista
    titulo, lat, long, loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig,driver=data_extractor_selenium(driver,restaurante["urls"],id,start)
    resultados.append({
            'Nombre': titulo, 'Localización': localidad, 'Latitud': lat, 'Longitud': long,
            'Dirección': loc, "Teléfono": telefono, "Precio": precios, 'Horario': horario,
            "Tipos de cocina": tipo_cocina, "Valoración": valora, "Número de votos": votos, 
            "Características": caracteristicas, "Descripción": descripcion, "url":restaurante["urls"],
            'url_web':enlace_web,'url_ig':enlace_ig
    })
    if lat=='KO':
        print(f'\033[91mCaptcha\033[0m')
        break
driver.quit()

df_resultados = pd.DataFrame(resultados)
df_resultados.to_csv('Data/Data_used/restaurant_data.csv',index=False,header=False,mode='a')
print(f'finalizado en {round((time.time()-start_0)/60,2)}min')