# Palindromos-Py 

Una librería de Python ligera y robusta para el análisis, verificación y generación de palíndromos. 

Esta librería no solo verifica si una palabra es un palíndromo, sino que también normaliza textos complejos (ignorando tildes y signos) y puede construir palíndromos a partir de un conjunto desordenado de letras.

## Características

- **Limpieza:** Elimina automáticamente tildes, diéresis, espacios y signos de puntuación.
- **Verificación de Viabilidad:** Determina si un conjunto de letras puede formar un palíndromo (basado en frecuencias de caracteres).
- **Generación:** Construye un palíndromo válido a partir de cualquier combinación de letras permitida.

## Estructura del Proyecto

```text
mi_libreria/
├── __init__.py    # Punto de entrada del paquete
├── core.py        # Lógica de verificación y construcción
└── utils.py       # Utilidades de procesamiento de texto

Uso Rapido

1.Limpieza de texto

from mi_libreria import limpiar_texto

texto = "¡Átate, demostráis!"
print(limpiar_texto(texto)) 
# Salida: 'atatedemostrais'

2.Verificacion
from mi_libreria import puede_ser_palindromo

resultado = puede_ser_palindromo("aabbc")
print(resultado) 
# Salida: 'Esta combinación puede formar un palíndromo'

3. Construir palindromo
from mi_libreria import construir_palindromo

print(construir_palindromo("aabbc")) 
# Salida: 'abcba'

Autor: Alejandro Del Rosal