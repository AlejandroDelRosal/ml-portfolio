from collections import Counter
from .utils import limpiar_texto

def puede_ser_palindromo(caracteres:str):
    """
    Verifica si existe una permutación de los caracteres que forme un palíndromo.

    Aplica un análisis de frecuencias basado en el principio de que un palíndromo
    solo puede tener, como máximo, un carácter con frecuencia impar.

    Args:
        caracteres (str): El texto o conjunto de caracteres a evaluar.

    Returns:
        str: Un mensaje indicando si es posible o no formar el palíndromo.
    
    Note:
        Para obtener un resultado booleano (True/False), podrías considerar 
        refactorizar esta función en el futuro.
    """
    texto_procesado= limpiar_texto(caracteres)

    conteo= Counter(texto_procesado)
    impares = [char for char, freq in conteo.items() if freq %2 !=0]

    if len(impares) > 1:
        return 'No es posible formar un palíndromo'
    return 'Esta combinacion puede formar un palindromo'

def construir_palindromo(caracteres:str) -> str:
    """
    Toma una cadena de caracteres y construye un palíndromo válido si es posible.

    La función analiza las frecuencias de cada letra. Si más de un carácter 
    aparece un número impar de veces, no se puede formar un palíndromo 
    y se retorna un mensaje de error.

    Args:
        caracteres (str): La cadena de texto o letras para procesar.

    Returns:
        str: El palíndromo formado o un mensaje indicando que es imposible.

    Example:
        >>> construir_palindromo("aabbc")
        'abcba'
    """
    texto= limpiar_texto(caracteres)
    conteo= Counter(texto)

    impares= [char for char, freq in conteo.items() if freq %2 !=0]
    if len(impares) > 1:
        return 'No es posible formar un palindormo con estos caracteres'
    mitad_izquierda = []
    caracter_central = ""

    # Distribuimos los caracteres
    for char, freq in conteo.items():
        # Si la frecuencia es impar, ese carácter (o uno de ellos) va al centro
        if freq % 2 != 0:
            caracter_central = char
        
        # Agregamos la mitad de las apariciones a la parte izquierda
        mitad_izquierda.append(char * (freq // 2))

    # Armamos el resultado
    izquierda = "".join(mitad_izquierda)
    derecha = izquierda[::-1] # La derecha es el espejo de la izquierda
    
    return izquierda + caracter_central + derecha
    