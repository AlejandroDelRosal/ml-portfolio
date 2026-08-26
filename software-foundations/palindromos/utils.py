import unicodedata
import re

def limpiar_texto(texto: str) -> str:
    """
    Limpia una cadena de texto para facilitar el análisis de palíndromos.

    Realiza tres pasos:
    1. Convierte a minúsculas.
    2. Elimina acentos y diéresis usando normalización NFKD.
    3. Elimina cualquier carácter que no sea alfanumérico (espacios, signos, etc.).

    Args:
        texto (str): La cadena original, posiblemente con tildes, mayúsculas o signos.

    Returns:
        str: Una cadena "plana" que contiene solo letras y números.

    Example:
        >>> limpiar_texto("¡Átate, demostráis!")
        'atatedemostrais'
    """
    texto = texto.lower() #convertimos a minusculas

    #eliminamos acentos y diéresis
    texto= unicodedata.normalize('NFKD', texto)
    texto=''.join([c for c in texto if not unicodedata.combining(c)])

    #Quitamos todo lo que no sea una letra
    texto= re.sub(r'[^a-z0-9]','',texto)

    return texto
