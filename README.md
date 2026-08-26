# ML Portfolio

Vitrina de modelos de machine learning. Este repo va a ir creciendo a lo largo del año con nuevos ejemplos (fraude, forecasting, recomendadores, visión por computadora, NLP con transformers, tracking de experimentos). Parte del [portafolio híbrido](https://github.com/AlejandroDelRosal/Portafolio) de Manuel Alejandro Del Rosal.

## Contenido actual

- `credit-risk-scoring/`: pipeline de ML para predicción de default (Random Forest, class weighting para datasets desbalanceados, feature importance para explicabilidad regulatoria)
- `software-foundations/`: fundamentos de ingeniería de software que sostienen los proyectos de arriba
  - `geometry_toolkit.py`: motor de geometría 2D/3D orientado a objetos (herencia, cálculo vectorial)
  - `palindromos/`: librería de procesamiento de strings (normalización Unicode, algoritmos O(n))

## Roadmap

- [ ] Detección de fraude
- [ ] Forecasting de series de tiempo
- [ ] Sistema de recomendación
- [ ] Clasificador de visión por computadora (CNN)
- [ ] Fine-tuning de un modelo de NLP (transformers)
- [ ] Tracking de experimentos con MLflow

## Convenciones

`main` protegido, todo por PR, conventional commits, CI con lint y tests. Ver [CHANGELOG.md](CHANGELOG.md).

## Licencia

MIT. Ver [LICENSE](LICENSE).
