# Laboratorio 04: El Perceptrón Multicapa (MLP) y Backpropagation desde Cero

**Curso:** SIS070 - Inteligencia Artificial  
**Unidad:** Unidad II - Aprendizaje Supervisado y Redes Neuronales  
**Estudiante:** Hector Armando Curo Lima  
**Repositorio:** [sis070-lab02-mlp-curo](https://github.com/HectorArmandoCuroLima/sis070-lab02-mlp-curo)

---

## 📖 Descripción del Proyecto

Este repositorio contiene la implementación desde cero de un Perceptrón Multicapa (MLP) utilizando **Python puro** y **NumPy**, sin depender de librerías de alto nivel como TensorFlow, Keras o PyTorch para la retropropagación. 

El objetivo principal es comprender la anatomía matemática de una red neuronal, implementando la propagación hacia adelante (Forward Propagation) y la retropropagación (Backpropagation) mediante la regla de la cadena y el descenso de gradiente. El modelo se entrena para resolver el clásico problema no lineal **XOR**.

## 📂 Estructura del Repositorio

```text
sis070-lab02-mlp-curo/
├── src/
│   └── mlp_implementation.py   # Código fuente del MLP (POO)
├── README.md                   # Documentación del laboratorio
└── requirements.txt            # Dependencias del proyecto


Actividad 1: Modificación de Hiperparámetros (Learning Rate)
Se probaron valores extremos de la tasa de aprendizaje con la función ReLU y una capa oculta durante 1000 épocas.

Learning Rate = 0.9:

Resultado: La pérdida bajó rápidamente al inicio (de 0.250 a 0.167) pero luego se estancó. Las predicciones fueron [[0.66], [0.66], [0.66], [0.01]].

Análisis: Un learning rate demasiado alto hace que el descenso de gradiente dé pasos gigantes, saltando por encima del mínimo global y quedándose atrapado en un mínimo local.

Learning Rate = 0.0001:

Resultado: La pérdida se mantuvo prácticamente constante (0.250081 a 0.250076). Las predicciones fueron [[0.50], [0.49], [0.49], [0.49]] (no aprendió).

Análisis: Un learning rate demasiado bajo hace que los pasos de actualización sean microscópicos. La convergencia es extremadamente lenta y el entrenamiento se vuelve inviable en un número razonable de épocas.

![alt text](image.png)

Actividad 2: Cambio de Función de Activación (ReLU vs. Sigmoide)
Se compararon ambas funciones en la capa oculta. Importante: Se corrigió la inicialización de pesos de * 0.1 a * 1.0 para evitar que las neuronas operaran en zonas planas de la Sigmoide.

ReLU con LR=0.1 y 1000 épocas:

Resultado: Pérdida final: 0.026. Predicciones: [[0.18], [0.90], [0.90], [0.08]].

Análisis: ReLU converge rápido y bien. Su derivada es 1 para valores positivos, permitiendo que el gradiente fluya sin desvanecerse.

Sigmoide con LR=0.1 y 1000 épocas:

Resultado: Pérdida final: 0.247. Predicciones: [[0.47], [0.52], [0.50], [0.53]].

Análisis: Con un LR bajo, la Sigmoide sufre de desvanecimiento del gradiente (su derivada máxima es 0.25).

Sigmoide con LR=1.0 y 5000 épocas:

Resultado: Pérdida final: 0.000215. Predicciones: [[0.01], [0.98], [0.98], [0.02]].

Análisis: Conclusión clave: Sigmoide SÍ puede resolver XOR, pero requiere una tasa de aprendizaje mucho mayor y más épocas. El desvanecimiento del gradiente se puede mitigar ajustando hiperparámetros.

![alt text](image-1.png) ![alt text](image-2.png)
Actividad 3: Ampliación Arquitectural (2 capas ocultas)
Se amplió el MLP a 4 capas en total (Entrada -> Oculta 1 -> Oculta 2 -> Salida).

MLP con 2 capas ocultas y ReLU (LR=0.5, 5000 épocas):

Resultado: Pérdida final: 0.250000 (no aprendió). Predicciones: [[0.5], [0.5], [0.5], [0.5]].

Análisis: Ocurrió el fenómeno de la "Neurona Moribunda" (Dying ReLU). Con una tasa de aprendizaje alta y una red más profunda, algunas neuronas recibieron actualizaciones que las llevaron a un estado permanentemente negativo. Su gradiente se volvió 0, congelando el aprendizaje de toda la red.

MLP con 2 capas ocultas y Sigmoide (LR=1.0, 5000 épocas):

Resultado: Pérdida final: 0.000085 (aprendizaje casi perfecto). Predicciones: [[0.006], [0.991], [0.991], [0.012]].

Análisis: La Sigmoide, aunque propensa al desvanecimiento del gradiente, no sufre de "muerte" neuronal como ReLU. Con un LR alto y suficiente tiempo, logró converger exitosamente en una arquitectura más profunda.

![alt text](image-3.png)![alt text](image-4.png) 

Transformar el MLP en uno de 4 capas (Entrada → Oculta1 → Oculta2 → Salida) y analizar su comportamiento.
![alt text](image-5.png) ![alt text](image-6.png)