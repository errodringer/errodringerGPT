# 🧠 ErrGPT en Español — Un Transformer desde Cero

Un **pequeño modelo GPT** entrenado con El Quijote, construido desde cero con PyTorch.
Pensado para ser **didáctico y educativo**, con arquitectura de Transformer completamente personalizable.

**Características principales:**
- 🎓 Código bien documentado y explicado paso a paso
- 🔧 Arquitectura Transformer modificable (capas, cabezas, dimensiones)
- 📊 Múltiples modelos entrenados con diferentes configuraciones
- 💾 Sistema de checkpoints y visualización de pérdida
- 🚀 Soporte para CPU y GPU (Apple Metal, CUDA)

---

## 📁 Estructura del proyecto

```
errodringer_gpt/
│
├── step1_get_data_quijote.py       ← Descarga y procesa El Quijote
├── step1_get_data_wikipedia.py     ← Descarga datos de Wikipedia
├── step2_model.py                  ← Arquitectura Transformer desde cero
├── step3_train.py                  ← Bucle de entrenamiento completo
├── step4_predict.py                ← Generación de texto e visualización
│
├── data/
│   ├── quijote.txt                 ← Texto sin procesar del Quijote
│   ├── vocabulary.json             ← Mapeo carácter ↔ índice
│   └── vocabulario.json            ← Mapeo alternativo
│
├── model/
│   ├── errgpt_*.pt                 ← Modelos entrenados (múltiples variantes)
│   ├── minigpt_*.pt                ← Modelos compactos entrenados
│   ├── history_loss.json           ← Historial de pérdida de entrenamiento
│   └── historial_perdida.json      ← Historial alternativo
│
└── requirements.txt                ← Dependencias del proyecto
```

---

## 🚀 Instalación

```bash
# Clonar el repositorio
git clone <tu-repo>
cd errodringer_gpt

# Crear entorno virtual (recomendado)
python3 -m venv .venv
source .venv/bin/activate  # En macOS/Linux
# o: .venv\Scripts\activate  # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

**Requisitos:**
- Python 3.8+
- PyTorch 2.0+
- NumPy, Matplotlib (para visualización)

> 💡 **No necesitas GPU.** Un portátil moderno puede entrenar en ~10-30 minutos con la configuración por defecto.
> Soporta **Apple Metal (MPS)**, **CUDA (NVIDIA)** y **CPU**.

---

## ▶️ Ejecución paso a paso

```bash
# Paso 1: Descargar y procesar El Quijote
python step1_get_data_quijote.py

# Paso 2: Verificar que el modelo se construye correctamente
python step2_model.py

# Paso 3: ¡Entrenar! Verás la pérdida bajar y texto generado cada 200 pasos
python step3_train.py

# Paso 4: Visualizar gráficas y generar texto con el modelo
python step4_predict.py
```

---

## 🎛️ Configuración del Modelo

Los hiperparámetros se definen en la clase `Config` dentro de [step2_model.py](step2_model.py#L47):

| Parámetro          | Por defecto | Descripción                              |
|--------------------|-------------|------------------------------------------|
| `context_length`   | 128         | Tokens de contexto que el modelo ve      |
| `num_layers`       | 4           | Bloques Transformer apilados             |
| `num_heads`        | 4           | Cabezas de atención paralelas            |
| `embedding_dim`    | 128         | Dimensión de vectores internos           |
| `batch_size`       | 384         | Secuencias procesadas simultáneamente    |
| `learning_rate`    | 1e-3        | Tasa de aprendizaje del optimizador      |
| `max_iterations`   | 5000        | Pasos totales de entrenamiento          |
| `dropout`          | 0.1         | Regularización (previene overfitting)    |

### Presets sugeridos:

**Para portátil lento (entrenamiento ~5-10 min):**
```python
context_length = 64
embedding_dim  = 64
num_layers     = 2
batch_size     = 32
max_iterations = 1500
```

**Para portátil estándar (entrenamiento ~15-30 min):**
```python
# Usar los valores por defecto de Config
```

**Para resultados de mejor calidad (entrenamiento ~1-2 horas):**
```python
context_length = 256
embedding_dim  = 256
num_layers     = 6
num_heads      = 8
batch_size     = 256
max_iterations = 10000
```

---

## 🧠 Conceptos Clave de la Arquitectura

### ¿Qué aprende el modelo?

El modelo aprende a **predecir el siguiente carácter** dado el contexto anterior.

**Ejemplo:** Si le muestras la secuencia `"En un lugar de la Manch"`, el modelo aprende que probablemente viene el carácter `"a"`.

Este proceso se repite miles de veces con diferentes fragmentos del texto hasta que el modelo entiende los patrones del idioma español.

### ¿Qué es la pérdida (Loss)?

La **pérdida** es una medida matemática del error del modelo:
- **Inicio del entrenamiento:** ~4.5 (aproximadamente log del tamaño del vocabulario, predicción aleatoria)
- **Durante el entrenamiento:** Baja gradualmente conforme el modelo aprende
- **Fin:** ~1.5-2.0 es un resultado decente para este proyecto

**Interpretación:**
- Si **solo la pérdida de entrenamiento baja** → el modelo está memorizando (overfitting)
- Si **ambas pérdidas (entrenamiento y validación) bajan** → el modelo está aprendiendo realmente

### Arquitectura Transformer

El modelo implementa un **Transformer Decoder** (similar a GPT):

```
Texto de entrada
      ↓
[Token Embedding]          ← Convierte caracteres en vectores densos
      ↓
[Positional Embedding]     ← Añade información de posición
      ↓
[Transformer Block] × N    ← El "cerebro" (N capas apiladas)
  ├─ [Multi-Head Attention] ← "¿Qué tokens anteriores son relevantes?"
  └─ [Feed-Forward Network]  ← Procesa la información de atención
      ↓
[Normalización]            ← Estabiliza el entrenamiento
      ↓
[Linear Layer]             ← Convierte vectores a probabilidades
      ↓
Probabilidades de cada carácter
```

### Multi-Head Attention

La **atención** es el corazón del Transformer. Permite que cada token:
1. "Mire hacia atrás" en tokens anteriores
2. Decida cuáles son relevantes para predecir el siguiente

**"Multi-head"** significa que hacemos esto varias veces en paralelo (`num_heads`), cada una aprendiendo a enfocarse en diferentes aspectos del texto.

**"Causal"** significa que un token NO puede ver el futuro — esto es crucial para que el modelo aprenda a predecir.

---

## 📊 Modelos Entrenados

El repositorio contiene varios modelos pre-entrenados con diferentes configuraciones:

| Modelo | Vocab | Context | Dim | Capas | Cabezas | Pasos | Tamaño |
|--------|-------|---------|-----|-------|---------|-------|--------|
| `minigpt_5000_256_8_8_256_128_1000.pt` | 5K | 256 | 256 | 8 | 8 | 1K | |
| `minigpt_30000_64_4_4_64_64_5000.pt` | 30K | 64 | 64 | 4 | 4 | 5K | |
| `minigpt_50000_128_4_4_128_128_10000.pt` | 50K | 128 | 128 | 4 | 4 | 10K | |
| `minigpt_100000_128_4_4_128_128_5000.pt` | 100K | 128 | 128 | 4 | 4 | 5K | |
| `errgpt_10000_128_4_4_128_384_5000.pt` | 10K | 128 | 128 | 4 | 4 | 5K | |

**Nota:** Puedes cargar cualquiera de estos modelos en [step4_predict.py](step4_predict.py) para generar texto.

---

## 🎓 Generación de Texto

Una vez entrenado, el modelo puede generar texto de forma **autoregresiva**:

1. Comienza con un **prompt** (ej: `"En un lugar de la Mancha"`)
2. Predice el siguiente carácter basado en el contexto
3. Añade ese carácter al contexto
4. Repite hasta generar la longitud deseada

**Parámetros que controlan la generación:**

- **Temperature:** Controla la "creatividad"
  - `0.3` → Conservador, repite frases conocidas
  - `1.0` → Equilibrio entre coherencia y variedad  
  - `2.0` → Muy creativo, pero puede ser incoherente

- **Top-K:** Solo considera los K caracteres más probables
- **Top-P:** Solo considera caracteres cuya probabilidad acumulada es ≤ P

---

## 🔧 Personalización Avanzada

### Cambiar el dataset

Puedes entrenar con otros textos. Simplemente crea un archivo de texto en `data/` y modifica [step1_get_data_quijote.py](step1_get_data_quijote.py) para cargarlo:

```python
TEXT_FILE = "data/tu_texto.txt"

# O cargar de una URL
URL = "https://ejemplo.com/texto.txt"
```

### Ajustar el vocabulario

El vocabulario está limitado a caracteres. Puedes modificar esta parte en [step1_get_data_quijote.py](step1_get_data_quijote.py) para usar **tokens BPE** o **palabras completas**.

### Entrenar con Wikipedia

El proyecto incluye [step1_get_data_wikipedia.py](step1_get_data_wikipedia.py) para descargar y procesar datos de Wikipedia:

```bash
python step1_get_data_wikipedia.py
python step3_train.py
```

Esto proporciona más texto diverso que El Quijote.

### Parámetros de Temperatura

Controla la "creatividad" en la generación:
- `0.3` → Conservador, repite frases conocidas
- `0.8` → Equilibrado (recomendado)
- `1.5` → Caótico, inventivo, a veces sin sentido

### ¿Por qué usamos El Quijote?
- Dominio público ✅
- En español ✅
- ~2M caracteres, suficiente para aprender patrones ✅
- El resultado tiene ese sabor cervantino que mola mucho en el vídeo ✅

---

## 📊 ¿Qué resultados esperar?

Después de entrenar el modelo con los parámetros por defecto:

**Iteración 100:** Pérdida ~3.5 (el modelo está aprendiendo patrones básicos)

**Iteración 1000:** Pérdida ~2.3 (puede generar estructuras de palabras similares al español)

**Iteración 5000:** Pérdida ~1.8 (genera frases más coherentes con el estilo del Quijote)

### Ejemplo de texto generado:

```
En un lugar de la Mancha, de que el caballero de los leones
que se había de hacer el escudero, que no se podía tener
en el camino de su amo, y así lo que había de...
```

¡No es perfecto, pero se nota que aprendió castellano! 🎉 Los patrones de escritura cervantina son reconocibles.

---

## 🎓 Detalles de Implementación

### Optimizador

Usamos **Adam** con:
- Learning rate: 1e-3 (ajustable)
- Beta1, Beta2: valores estándar
- Weight decay: para regularización

### Funciones de Pérdida

- **Cross-Entropy Loss** para clasificación de caracteres
- Calculada en todos los tokens excepto el primero

### Regularización

- **Dropout:** Desactiva aleatoriamente neuronas durante entrenamiento
- **Weight decay:** Penaliza pesos grandes
- **Validación periódica:** Previene overfitting

### Checkpoints

Los modelos se guardan con su configuración completa:

```python
checkpoint = {
    "model_state": model.state_dict(),
    "config": config.__dict__,
    "vocabulary": {
        "char_to_idx": char_to_idx,
        "idx_to_char": idx_to_char,
    },
    "iteration": iter,
    "train_loss": train_loss,
}
```

---

## 🐛 Troubleshooting

**"Can't find 'data/vocabulary.json'"**
→ Ejecuta `step1_get_data_quijote.py` primero

**"CUDA out of memory"**
→ Reduce `batch_size` o `embedding_dim` en Config

**"El modelo no mejora (pérdida se estanca)"**
→ Reduce `learning_rate` o aumenta `max_iterations`

**"La salida es basura/aleatoria"**
→ El modelo necesita más iteraciones de entrenamiento

---

## 📚 Referencias y Recursos

- **nanoGPT** de Andrej Karpathy: https://github.com/karpathy/nanoGPT
- **"Attention is All You Need"** (Vaswani et al., 2017) — El paper original del Transformer
- **The Illustrated Transformer** — Jay Alammar: https://jalammar.github.io/illustrated-transformer/
- **PyTorch Documentation:** https://pytorch.org/docs/stable/index.html

### Para escalar este proyecto:

1. **Fine-tuning:** Usar modelos pre-entrenados (GPT-2, Llama) como base
2. **LoRA:** Low-Rank Adaptation para entrenar eficientemente
3. **Hugging Face:** Integración con transformers library
4. **Quantization:** Comprimir modelos para inferencia más rápida

---

## 📄 Licencia

Este proyecto usa El Quijote, que está en dominio público.

---

## 🙌 Agradecimientos

Inspirado en **nanoGPT** de Andrej Karpathy y la arquitectura Transformer original.

**¿Te ha gustado?** Considera dejar una ⭐ en el repositorio.

---

*Última actualización: Febrero 2026*
