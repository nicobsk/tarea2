
# Proyecto 2 - Clasificación de clima (Weather Classification) 

Introducción a la Inteligencia Artificial 

## Integrantes

- Nicolás Mena Saavedra

## Descripción del problema

El objetivo de este proyecto es clasificar imágenes de condiciones climáticas en 4 categorías:
**Cloudy** (nublado), **Rain** (lluvia), **Shine** (soleado) y **Sunrise** (amanecer), usando un modelo de Deep Learning pre-entrenado (transfer learning). 
Es un problema de clasificación de imágenes multiclase, relevante como ejercicio práctico de visión computacional con un dataset de tamaño acotado, típico de escenarios donde no se cuenta con millones de imágenes etiquetadas para entrenar un modelo desde cero.

## Dataset y fuente

- **Fuente:** [Multi-class Weather Dataset](https://www.kaggle.com/datasets/pratik2901/multiclass-weather-dataset) (Kaggle).
- **Tamaño original:** 1125 imágenes distribuidas en 4 clases (Cloudy: 300, Rain: 215, Shine: 253, Sunrise: 357).
- **Tamaño final:** 1019 imágenes, tras eliminar 106 imágenes casi-duplicadas dentro de la misma clase (detectadas vía perceptual hashing), para evitar data leakage entre los splits de entrenamiento/validación/test.
- **Estructura esperada** (no incluida en el repositorio por su peso, ver instrucciones de descarga más abajo):

```
data/weather/
├── Cloudy/
├── Rain/
├── Shine/
└── Sunrise/
```

### Cómo obtener el dataset

1. Descargar el "Multi-class Weather Dataset" desde Kaggle.
2. Descomprimir y ubicar las 4 carpetas de clases (`Cloudy`, `Rain`, `Shine`, `Sunrise`)   directamente dentro de `data/weather/` en la raíz del proyecto (sin carpeta intermedia).

## Justificación del modelo

Se utilizó **transfer learning con ResNet50** (pesos preentrenados en ImageNet), congelando el backbone convolucional y reentrenando solo una capa final adaptada a las 4 clases del problema.

- **¿Por qué transfer learning?** Con un dataset de aprox. 1000 imágenes, entrenar una red profunda desde cero llevaría a overfitting severo. Transfer learning permite aprovechar representaciones visuales generales ya aprendidas por ResNet50 sobre millones de imágenes.
- **¿Por qué ResNet50?** Buen balance entre capacidad de representación y costo computacional, ampliamente documentada y disponible directamente en `torchvision` con pesos de ImageNet.
- **Alternativas consideradas:** MobileNetV2/EfficientNetB0 (más livianas, pero se priorizó la capacidad dado el acceso a GPU dedicada).

## Metodología

1. **EDA:** Conteo de imágenes por clase (desbalance moderado, ratio 0.6), verificación de integridad de archivos (sin corruptos), análisis de dimensiones (506x335 px promedio) inspección visual por clase, y detección de duplicados (eliminación de 106 near-duplicates dentro de la misma clase).

2. **Split de datos:** 70% train / 15% val / 15% test, estratificado por clase.

3. **Preprocesamiento y feature engineering:** Resize a 224x224, normalización con estadísticas de ImageNet, y data augmentation en train (flip horizontal, rotación ±15°, color jitter).
4. **Modelo:** ResNet50 preentrenada, backbone congelado, capa final reemplazada por   `Dropout(0.5) + Linear(2048, 4)`.

5. **Entrenamiento:** Optimizador AdamW (lr=0.001, weight_decay=0.2), `CrossEntropyLoss`, early stopping con patience=4 sobre el accuracy de validación.

6. **Evaluación:** Classification report, matriz de confusión y F1 macro, calculados tanto en validación (seguimiento durante desarrollo) como en test (métrica oficial final).

## Resultados

El entrenamiento convergió en 13 epochs (early stopping), con el mejor checkpoint guardado en la epoch 9 (val accuracy: 93.4%). Evaluado sobre el **set de test** (independiente, nunca usado para entrenar ni para seleccionar el checkpoint):

| Métrica | Valor |
|---|---|
| Accuracy | 96% |
| F1 macro | 0.95 |

| Clase | Precision | Recall | F1-score |
|---|---|---|---|
| Cloudy | 0.95 | 0.97 | 0.96 |
| Rain | 1.00 | 0.97 | 0.98 |
| Shine | 0.89 | 0.91 | 0.90 |
| Sunrise | 0.98 | 0.96 | 0.97 |

La matriz de confusión mostró que **Shine** es la clase más propensa a errores, mezclándose tanto con Cloudy como con Sunrise, mientras que **Rain** obtuvo el mejor desempeño consistente gracias a su señal visual distintiva (agua/gotas visibles). Dos pruebas adicionales con imágenes completamente externas al dataset (escenas lluviosas urbanas) fueron clasificadas correctamente con alta confianza, confirmando la capacidad de generalización del modelo.

## Conclusiones

El modelo alcanzó un desempeño sólido (96% accuracy en test) usando transfer learning sobre un dataset relativamente pequeño, sin señales de overfitting gracias a las estrategias de regularización aplicadas (Dropout, weight decay, data augmentation). La principal limitación que se detectó fue que Cloudy y Shine pueden confundirse, ya que una misma escena (cielo parcialmente nublado con sol visible) puede pertenecer a cualquiera de las dos categorías.
**Posible trabajo futuro:** Se buscará fine-tuning de capas convolucionales adicionales del backbone, balanceo de clases más explícito, y ampliar el dataset con más ejemplos ambiguos de Cloudy/Shine/Sunrise para mejorar el modelo.

## Estructura del repositorio

```
tarea2/
├── README.md
├── environment.yml
├── weather_classification.ipynb
└── data/              <- no incluido en el repo, ver "Cómo obtener el dataset"
```

## Cómo reproducir

```bash
conda env create -f environment.yml
conda activate introalaia
jupyter notebook weather_classification.ipynb
```
