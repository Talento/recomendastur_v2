import re
import json
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")
######################################50#############################################################################
################################            RESTAURANTES            ######################################################
###################################################################################################################
def process_formato_1(soup):
############# location #########################
    loc_element=soup.find('div',{'id':'info_location'})
    loc=loc_element.find_all('div')[1].text.replace('\n','').strip()
    ############ localidad #################
    localidad=loc.split(',')[-3]
    ############ telefono ~######################
    try:
        telefono = soup.find('a', href=re.compile(r'^tel:')).get('href').replace('tel:','')
    except:
        telefono='Desconocido'
    ############# precios #####################
    try:
        precios=soup.find('span',{'class':'nowrap'}).text
    except:
        precios='Desconocido'
    ############ horario ###################
    try:
        horario=soup.find('table',{'class':'schedule-table'}).text.split('\n')
        horario=[item for item in horario if item!='']
    except:
        horario='Desconocido'
    ########### Tipos de cocina	 ##################
    try:
        tipo_cocina_element=soup.find('div',{'class':'cuisine_wrapper'})
        tipo_cocina_span=tipo_cocina_element.find_all('span')
        tipo_cocina=[tipo.text.replace('\n','').strip() for tipo in tipo_cocina_span]
    except:
        tipo_cocina='Desconocido'
    ############ Valoración ###############
    valora=soup.find('div',{'class':re.compile(r'stars__fill')}).get("style").replace('width:','').replace('%','')
    ######### Número de votos #################
    try:
        votos=soup.find('span',{'class':'rating-stars__text'}).text.replace('votos','').strip()
    except:#
        votos='Desconocido'
    ########## Características ##############
    try:
        features_block=soup.find('div',{'class':'features_block'})
        spans=features_block.find_all('span')
        caracteristicas=[cara.text for cara in spans]
    except:
        caracteristicas='Desconocido'
    ############## description #################
    #wrapper_description
    try:
        descripcion=soup.find('div',class_='description').text.replace('\n','')
    except:
        descripcion='Desconocido'
    ############## enlaces de interes #################
    ######## web personal
    try:
        element_web=soup.find('div',class_='website')
        enlace_web=element_web.find('a').text.strip()
    except:
        enlace_web='Desconocido'
    ######## ig personal
    try:
        element_ig=soup.find('div',class_='instagram')
        enlace_ig=element_ig.find('a').text.strip()
    except:
        enlace_ig='Desconocido'        
    return loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig
def process_formato_2(soup):
    ############# location #########################
    loc=soup.find('div',class_='sidebar-address__info').text.replace('\n','').strip()
    ############ localidad #################
    localidad=loc.split(',')[-4]
    ############ telefono ~######################
    try:
        telefono = soup.find('a', href=re.compile(r'^tel:')).get('href').replace('tel:','')
    except:
        telefono='Desconocido'
    ############# precios #####################
    try:
        precios_div=soup.find('div',{'class':'price__info'})
        precios=precios_div.find_all('span')[-1].text
    except:
        precios='Desconocido'
    ############ horario ###################
    try:
        horario_ele=soup.find_all('li',{'class':'schedule-week__item'})
        horario=[item.text.replace('\n',' ').strip() for item in horario_ele if item.text!='']
    except:
        horario='Desconocido'
    ########### Tipos de cocina	 ##################
    try:
        tipo_cocina_element=soup.find('div',{'class':'sidebar-details__cuisine'})
        tipo_cocina_span=tipo_cocina_element.find_all('span')
        tipo_cocina=[tipo.text.replace('\n','').strip() for tipo in tipo_cocina_span]
    except:
        tipo_cocina='Desconocido'
    ############ Valoración ###############
    valora=soup.find('div',{'class':re.compile(r'stars__fill')}).get("style").replace('width:','').replace('%','')
    ######### Número de votos #################
    try:
        votos=soup.find('span',{'class':'chip chip--tertiary values-number unclickable'}).text.replace('votos','').strip()
    except:#
        votos='Desconocido'
    ########## Características ##############
    try:
        spans=soup.find_all('span',class_=re.compile(r'sidebar-details__features-item'))
        caracteristicas=[cara.text.strip() for cara in spans]
    except:
        caracteristicas='Desconocido'
    ############## description #################
    #wrapper_description
    try:
        descripcion=soup.find('div',class_='description').text.replace('\n','')
    except:
        descripcion='Desconocido'
    ############## enlaces de interes #################
    ######## web personal
    try:
        element_web=soup.find('div',class_='sidebar-details__website')
        enlace_web=element_web.find('a').text.strip()
    except:
        enlace_web='Desconocido'
    ######## ig personal
    try:
        element_ig=soup.find('div',class_='sidebar-details__instagram')
        enlace_ig=element_ig.find('a').text.strip()
    except:
        enlace_ig='Desconocido'        
    return loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig
def from_responseContent_to_data(response,print_=False):
    """
    Función: 
        para obtener la información relevante de un determinado restaurante a partir de la respuesta (requests) de una página cargada de restaurantguru.
        Si no ebncuentra las coordenadas devuelve un KO en titulo para poder procesarlo luego

    Parámetros:
    - response: response.content de un request.get
    - print_: si quieres ver los datos obtenidos

    Salida:
    - nombre: nombre del restaurante correspondiente.
    - lat, long: coordenadas del restaurante.
    - loc: dirección completa del restaurante.
    - localidad: localidad (ciudad, pueblo, ...) en la que se encuentra el restaurante.
    - precios: rango de precio medio por persona o precio máximo por persona del restaurante (algunos son desconocidos).
    - telefono: contacto del restaurante (con prefijo).
    - horarios: horario del restaurante (en formato lista, cada día un elemento, ordenados de lunes a domingo)
    - tipo: tipos de cocina del restaurante (lista). 
    - descripcion: pequeña descripción del restaurante (su extensión y precisión depende del alojamiento, algunos no disponen de ella).
    - caracteristicas:  servicios o características que ofrece el restaurante (lista).
    
    """
    #start=time.time()
    soup = BeautifulSoup(response, 'html.parser')
    #print(f't parser: {round(time.time()-start,4)}')
    ############## nombre ###############
    try:
        title_elelment= soup.find('div', {'class': 'title_container'})
        titulo= title_elelment.find('h1', {'class': 'notranslate'}).text.strip()
        formato=1
    except:
        titulo=soup.find('h1',class_='title main-restaurant__title').text.strip()
        formato=2
    ######################## lat y lon ###########################
    try:
        script_tag = soup.find('script', {'type': 'application/ld+json'})
        coord = json.loads(script_tag.string)
        # Accede a la latitud y longitud
        lat = coord['geo']['latitude']
        long = coord['geo']['longitude']
    except:
        try:
            element=soup.find('a',class_='direction_link').get('href').split('=')[-1].split(',')
            # Accede a la latitud y longitud
            lat = element[0]
            long = element[1]
        except:
            try:
                element=soup.find('a',class_='btn btn--secondary sidebar-map__create-route--mobile').get('href').split('=')[-1].split(',')
                # Accede a la latitud y longitud
                lat = element[0]
                long = element[1]
            except:          
                return 'KO', 'lat', 'long', 'loc', 'localidad', 'precios', 'telefono', 'horario', 'tipo_cocina', 'valora','votos','caracteristicas', 'descripcion','',''
    ######################## EN FUNCION DEL FORMATO EXTRAE DATA ####################################
    if formato==1:
        loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig=process_formato_1(soup)   
    elif formato==2:
        loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig=process_formato_2(soup)  
    #Por si quieres hacer print de variables
    if print_==True:
        for item in [titulo, lat, long, loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig]:
            print(item)

    return titulo, lat, long, loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig
def data_extractor_selenium(driver,url,id,start):
    """
    Función: 
        extractor de datos especifico para restaurantes que usa selenium. Si hay error de coordenadas devuelve KO en titulo, si hay otro tipo de error devuelve KO en 
        latitud

    Parámetros:
    - driver
    - url
    -id : numero o indice de la url en el ddataframe
    -start: tiempo inicial para calcualr cuanto tarda 

    Salida:
        titulo, lat, long, loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion
    """
    html=driver.page_source
    try:
        titulo, lat, long, loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig=from_responseContent_to_data(html,print_=False)
        if titulo!='KO':
            print(f'OK({id}) en t={round(time.time()-start,4)}s URL: {url}')
        else:
            print(f"\033[33mError no se han obtenido coordenadas en la URL ({id}): {url}\033[0m")
        return titulo, lat, long, loc, localidad, precios, telefono, horario, tipo_cocina, valora,votos,caracteristicas, descripcion,enlace_web,enlace_ig,driver
    except:
        print(f'\033[91mKO({id}) URL: {url}\033[0m')
        return 'KO', 'KO', 'long', 'loc', 'localidad', 'precios', 'telefono', 'horario', 'tipo_cocina', 'valora','votos','caracteristicas', 'descripcion','','',driver
def manual_captcha_resolve(url):
    """
    Función: 
        dada una url cualquiera crea una bventana con selenium para que puedas acceder y resolver un captcha, luego de esto, deberas escribir el input,
        "y" si lo hAS resuelto o "n" si ha ocurrido algun error. Puedes usar "break" para finalizar el proceso.

    Parámetros:
    - url: str de una url

    Salida:
    - captcha: str, "y" o "n", "break para finalizar el proceso"
    """
    option = webdriver.ChromeOptions()
    driver = webdriver.Chrome(options=option)
    driver.get(url)
    captcha=input('Captcha resuelto?y/n   ')
    driver.quit()

    return captcha