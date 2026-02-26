# 🧠 MiniGPT en Español — Código para el vídeo de YouTube

Un GPT pequeñito entrenado con El Quijote, construido desde cero con PyTorch.
Pensado para ser **didáctico**, no para ganar benchmarks. 😄

---

## 📁 Estructura del proyecto

```
minigpt/
│
├── paso1_preparar_datos.py     ← Descarga y tokeniza El Quijote
├── paso2_modelo.py             ← El Transformer desde cero
├── paso3_entrenar.py           ← El bucle de entrenamiento
├── paso4_visualizar_y_jugar.py ← Gráficas y generación interactiva
│
├── datos/
│   ├── quijote.txt             ← Texto crudo (se descarga automáticamente)
│   ├── train.bin               ← Datos de entrenamiento (tokens)
│   ├── val.bin                 ← Datos de validación
│   └── vocabulario.json        ← Mapeo char ↔ índice
│
└── modelo/
    ├── minigpt_quijote.pt      ← El modelo entrenado
    ├── historial_perdida.json  ← Pérdida por iteración
    ├── curva_perdida.png       ← Gráfica de entrenamiento
    └── mapa_atencion.png       ← Visualización de atención
```

---

## 🚀 Instalación

```bash
pip install torch numpy matplotlib
```

> No necesitas GPU. Un portátil moderno puede entrenar el modelo
> en ~10-30 minutos con la configuración por defecto.

---

## ▶️ Ejecución paso a paso

```bash
# Paso 1: Preparar los datos (El Quijote desde Project Gutenberg)
python paso1_preparar_datos.py

# Paso 2 (opcional): Comprobar que el modelo se construye bien
python paso2_modelo.py

# Paso 3: ¡Entrenar! Verás la pérdida bajar y texto generado cada 200 pasos
python paso3_entrenar.py

# Paso 4: Ver gráficas y jugar con el modelo entrenado
python paso4_visualizar_y_jugar.py
```

---

## 🎛️ Ajustar el tamaño del modelo (en `paso2_modelo.py`)

| Parámetro       | Valor por defecto | ¿Qué hace?                        |
|----------------|-------------------|-----------------------------------|
| `n_capas`       | 4                 | Bloques Transformer apilados      |
| `n_cabezas`     | 4                 | Cabezas de atención               |
| `dim_embedding` | 128               | Tamaño de los vectores internos   |
| `long_contexto` | 64                | Tokens de contexto hacia atrás    |
| `max_iteraciones`| 3000             | Pasos de entrenamiento            |

**Para un portátil lento:**
```python
dim_embedding = 64
n_capas       = 2
max_iteraciones = 1500
```

**Para resultados más ricos (necesita más tiempo):**
```python
dim_embedding = 256
n_capas       = 6
max_iteraciones = 5000
```

---

## 🧠 Conceptos clave del vídeo

### ¿Qué aprende el modelo?
A predecir el siguiente **carácter** dado el contexto anterior.
Si le muestras `"En un lugar de la Manch"`, aprende que probablemente viene `"a"`.

### ¿Qué es la pérdida (loss)?
Una medida del error. Empieza alta (~4.5 = log del vocabulario, predicción aleatoria)
y baja conforme el modelo aprende (~1.5-2.0 es un resultado decente).

### ¿Qué es la temperatura?
Controla la "creatividad":
- `0.3` → conservador, repite frases conocidas
- `0.8` → equilibrado (recomendado)
- `1.5` → caótico, inventivo, a veces sin sentido

### ¿Por qué usamos El Quijote?
- Dominio público ✅
- En español ✅
- ~2M caracteres, suficiente para aprender patrones ✅
- El resultado tiene ese sabor cervantino que mola mucho en el vídeo ✅

---

## 📊 ¿Qué resultados esperar?

Después de 3000 iteraciones en CPU, el modelo generará cosas como:

```
En un lugar de la Mancha, de que el caballero de los leones
que se había de hacer el escudero, que no se podía tener
en el camino de su amo, y así lo que había de...
```

¡No es perfecto, pero se nota que aprendió castellano! 🎉

---

## 🔗 Referencias y para ir más lejos

- **nanoGPT** de Andrej Karpathy: https://github.com/karpathy/nanoGPT
- **"Attention is All You Need"** (el paper original del Transformer)
- Para escalar: fine-tuning con Hugging Face, LoRA, etc.

---

*Código creado para el canal de YouTube. Si te ha servido, ¡suscríbete! 🙌*
