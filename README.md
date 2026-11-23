# Clasificación de imágenes renales con una CNN y YOLO26n-cls

El proyecto compara cuatro clases por imagen: **Cyst, Normal, Stone y Tumor**. La CNN de
referencia es la del notebook público de Ahnaf Tahmeed, identificado en los notebooks previos.
El comparador es YOLO26n-cls de Ultralytics 8.4.0, con cabeza de cuatro clases.

## Repositorio

[Repositorio técnico del experimento](https://github.com/juanfernandojaimesperez63-source/comparacion-renal-cnn-yolo-publico)
(acceso público). Contiene código, preparación de datos, configuraciones,
resultados y documentación técnica.

## Estado del proyecto

**Completado y verificado el 22 de noviembre de 2026.** Autores: Juan Fernando
Jaimes Pérez, Miguel Andrés Veloza y Gabriela Contreras García. Afiliación: Programa de Ingeniería de Sistemas,
Universidad Simón Bolívar, sede Cúcuta, Colombia. Se ejecutaron cinco semillas por modelo,
diez entrenamientos y diez evaluaciones sobre las mismas 2.633 imágenes de prueba.
Todas las corridas definitivas utilizaron el backend compartido PyTorch 2.9.1 y
GPU Tesla T4 de Colab. Los pilotos locales se conservaron fuera de esta entrega.

| Resultado de prueba (media ± desviación estándar muestral) | CNN | YOLO26n-cls |
| --- | ---: | ---: |
| F1 macro | 0,7077 ± 0,0234 | 0,7478 ± 0,0537 |
| Exactitud | 0,7322 ± 0,0266 | 0,7676 ± 0,0538 |
| Latencia p50 (ms, media entre semillas) | 1,5233 ± 0,2536 | 6,3146 ± 0,1105 |

La diferencia media de F1 (YOLO menos CNN) fue 0,0400; la prueba bilateral exacta
pareada dio p = 0,1875. El mayor promedio observado no demuestra superioridad
estadística. La inferencia se limita a esta partición y a la variación entre semillas.

### Entregables

- `Articulo_IEEE.pdf`: ocho páginas de contenido más una de referencias; fuentes editables en `article/`.
- `results/consolidado.xlsx`: resumen, diez corridas, resultados por clase y prueba pareada, con fórmulas.
- `results/full/`: historiales, predicciones, métricas, curvas, confusiones y ejemplos de error por corrida.
- `results/verification.json`: comprobación independiente de métricas desde las predicciones.
- `notebooks/Comparacion_renal_Drive.ipynb`: notebook ejecutado con salidas de Colab. La copia pública conserva esas salidas y retira metadatos de sesión y un mensaje con la cuenta personal de Drive.
- `src/`, `data/` y `docs/`: código, auditoría y fundamentación de la selección de YOLO.

La CNN se adaptó desde Keras conservando capas, padding y 456.132 parámetros.
Al transferir los mismos pesos, la diferencia máxima de probabilidades fue
4,47e-8 (`docs/architecture_equivalence.json`). YOLO tiene 1.536.228 parámetros.

### Evidencias y CSV individuales

Los archivos `results/cnn-17.csv` a `results/cnn-57.csv` y `results/yolo-17.csv`
a `results/yolo-57.csv` conservan las métricas originales por ejecución.
`results/evidencias_por_corrida.zip` contiene las carpetas completas `audit/` y `full/`,
con historiales, configuraciones, predicciones y figuras. Se distribuyen comprimidas
en GitHub para facilitar su descarga; el ZIP académico ya las incluye extraídas.
El archivo no incluye imágenes originales del dataset ni pesos. Antes de verificar
o reproducir desde una descarga de GitHub, ejecuta:

```bash
python data/unpack_evidence.py
```

El script verifica SHA-256, rutas y contenido; conserva los archivos existentes
que coinciden y no sobrescribe archivos distintos. `results/evidencias_manifest.json`
guarda los hashes de todos los archivos. Si las carpetas ya están completas,
el script solo comprueba su integridad.

Para comprobar las cifras guardadas sin volver a entrenar:

```bash
python src/verify_results.py --manifest results/audit/manifest.csv --results results
```

La opción `--require-checkpoints` verifica además la presencia de los pesos locales.
El paquete de entrega conserva las predicciones y excluye los pesos para limitar su tamaño.

## Datos y procedencia

Kaggle declara **CC0** para la recopilación de GonzaJL (versión 2, 28 de octubre de 2023).
La página atribuye las imágenes a cuatro conjuntos. La API de Kaggle mostró las siguientes
licencias el 11 de noviembre de 2026. El CSV incluido vincula nombres de archivo con
etiquetas, sin identificar la fuente o modalidad por imagen.

| Fuente declarada | Licencia mostrada en Kaggle |
| --- | --- |
| [CT Kidney Dataset, Islam y Kabir](https://www.kaggle.com/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone) | CC BY 4.0 |
| [Kidney Stones MRI and CT scans](https://www.kaggle.com/datasets/mohammedrizwanmalik/kidney-stones-mri-and-ct-scans) | Unknown |
| [KidneyStone](https://www.kaggle.com/datasets/raagbhutani/kidneystone) | Unknown |
| [Medical Scan Classification Dataset](https://www.kaggle.com/datasets/arjunbasandrai/medical-scan-classification-dataset) | Other (specified in description). La descripción identifica otra recopilación y no detalla términos por imagen. |

Citar la recopilación y las fuentes declaradas. La etiqueta CC0 de la recopilación deja
pendiente verificar los derechos de reutilización de las imágenes de origen. El ZIP del
proyecto contiene código y resultados tabulares, sin el dataset ni pesos; sí incluye figuras de análisis con ejemplos del conjunto.

### Auditoría ejecutada

Se decodificaron **11.756 imágenes RGB de 224 × 224**, con **2.939 por clase**. Hay
**9.417 JPEG, 2.339 PNG y cero archivos dañados**. `datos.csv` contiene 11.756 rutas
distintas y todas sus etiquetas coinciden con las carpetas. El formato varía según
la clase. Por ejemplo, todos los quistes y tumores son JPEG y parte de las clases
normal y cálculo son PNG. Esta asociación y la mezcla de fuentes pueden facilitar
atajos visuales por protocolo de adquisición o compresión.

El análisis identificó **1.225 copias exactas adicionales** en **1.222 grupos de
duplicados**. En esta versión el conteo coincide al comparar los bytes y los píxeles
RGB. Para buscar similitud entre imágenes con píxeles distintos se exigió distancia
de Hamming ≤ 4 en un dHash de 64 bits y RMSE ≤ 0,035 entre miniaturas grises de
32 × 32 normalizadas a [0, 1]. El segundo criterio redujo el encadenamiento de
vecinos del dHash solo. El archivo `similar_pairs.csv` conserva **21.106 pares de píxeles distintos que cumplen ambos umbrales**,
sin pares entre clases bajo ambos criterios. Se inspeccionaron visualmente 12 pares
cerca del umbral de RMSE. La muestra permite inspeccionar el comportamiento del
filtro y carece de información clínica para identificar pacientes.

Se agruparon los pares por componentes conexos y se eliminaron las copias exactas
redundantes. Se asignaron los grupos completos con semilla 27, ajustando por clase
al objetivo de la guía. No hubo grupos con etiquetas en conflicto con esta regla.

| Clase | Entrenamiento | Prueba | Validación | Copias retiradas |
| --- | ---: | ---: | ---: | ---: |
| Cyst | 1.879 | 671 | 134 | 255 |
| Normal | 1.937 | 692 | 138 | 172 |
| Stone | 1.499 | 535 | 107 | 798 |
| Tumor | 2.057 | 735 | 147 | 0 |
| **Total** | **7.372** | **2.633** | **526** | **1.225** |

La partición final es 70,00 % entrenamiento, 25,00 % prueba y 4,99 % validación
(porcentajes aproximados). El CSV, los nombres secuenciales y los metadatos EXIF
no proporcionan identificadores de paciente verificables. El agrupamiento por
similitud limita la fuga visual bajo los umbrales elegidos. Puede haber cortes del
mismo paciente que el método no detecte, por ejemplo al cambiar de plano.

## Reproducción y protocolo

| Parámetro | CNN | YOLO26n-cls |
| --- | --- | --- |
| Arquitectura | 4 bloques Conv2D/MaxPool, Flatten, Dense lineal de 128, salida softmax | YAML YOLO26n-cls, cabeza ajustada a 4 clases |
| Parámetros verificados | 456.132 | 1.536.228 |
| Framework | PyTorch 2.9.1 (adaptación de la arquitectura Keras) | PyTorch 2.9.1, Ultralytics 8.4.0 |
| Inicialización | Aleatoria, sin pesos externos | Aleatoria, sin pesos externos |
| Optimizador | Adam, 0,001, épsilon 1e-7 | Adam, 0,001, épsilon 1e-7 |
| Pérdida | Entropía cruzada sobre logits | Entropía cruzada sobre logits |
| Entrada y lote | RGB 224 × 224, lote 32 | RGB 224 × 224, lote 32 |
| Aumentos de entrenamiento | Giros uniformes de ±108°, volteos H/V con p=0,5 y relleno negro | Mismos aumentos y mismo orden por semilla |
| Selección | Mayor F1 macro de validación, máximo 20 épocas, paciencia 5 | Mismo criterio |
| Semillas | 17, 27, 37, 47, 57 | Mismas semillas |

La CNN conserva los filtros 32/64/128/256, núcleos 5/5/3/3, stride 2, padding
`same`, activaciones ReLU, MaxPool 2×2 y capas densas de la fuente. El notebook
histórico usó lote 64, validación del 25 %, relleno de rotación por reflexión y
checkpoint por exactitud de validación. Reportó `val_accuracy=0,87411` para el mejor
checkpoint en el registro de Keras. Para este experimento se fijaron la partición
compartida, lote, transformaciones y criterio de selección antes de abrir prueba.
La cifra histórica se cita como referencia de otro protocolo de validación.
La adaptación conserva la arquitectura. Frente al entrenamiento Keras histórico,
las inicializaciones y la implementación numérica de Adam pueden cambiar; la
comparación actual fija PyTorch como backend común para las dos redes.

El tiempo de entrenamiento mide pared desde el primer paso hasta la escritura del
último historial, con aumentos, validación y guardado. La latencia por imagen
usa lote 1, 20 calentamientos y 100 mediciones sobre entradas ya decodificadas
y redimensionadas. Se incluye normalización, transferencia cuando aplica, paso
por la red y devolución de probabilidades a CPU. Ejecutar los dos modelos en
el mismo tipo de GPU para comparar tiempos. Los tiempos de la comparación final deben interpretarse junto con el entorno de Colab
registrado en cada ejecución.

La prueba bilateral exacta invierte los signos de las cinco diferencias pareadas
de F1 macro y usa su media como estadístico. Requiere que los signos de las
diferencias sean intercambiables bajo la hipótesis nula. El conjunto de prueba fijo
hace que la inferencia sea condicional a esa partición. Con cinco pares hay 32
configuraciones, y el menor valor p bilateral posible es 0,0625. Reportar también
diferencias por semilla, media y desviación estándar muestral.

## Entorno reproducible

`requirements.txt` contiene las dependencias directas. `requirements-lock-colab.txt`
conserva las 83 dependencias activas y sus versiones del entorno final de Linux/Colab.
`results/dependencies.txt` es un inventario completo del entorno base de Colab,
incluidos paquetes ajenos al proyecto: no debe usarse como manifiesto de instalación.
Los pesos están conservados en la copia local de trabajo y en los resultados privados
de Drive; se excluyen del ZIP. Las predicciones permiten verificar las métricas sin pesos.

## Ejecutar con el backend compartido

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash data/download.sh ../datos
python data/audit.py ../datos/raw ../audit_reproducida
cmp ../audit_reproducida/manifest.csv results/audit/manifest.csv
python src/experiment.py --root ../datos/raw \
  --manifest results/audit/manifest.csv --output ../resultados_reproduccion \
  --backend shared_torch
```

El coordinador hace primero pruebas breves, completa diez entrenamientos, abre
prueba después de terminar todos y genera el consolidado. Usa una carpeta nueva
para reproducir sin alterar los resultados entregados. Un nuevo proceso por
corrida libera el estado del modelo y la memoria al terminar.

En cada llamada a `src/run.py`, usa `--backend shared_torch` para reproducir las
corridas definitivas. Este backend selecciona CUDA, MPS o CPU según disponibilidad.
Conserva el mismo dispositivo para ambos modelos. El backend `original` mantiene
la implementación Keras de referencia para contrastar la adaptación. No mezcles
resultados de ambos backends en una misma carpeta.

## Ejecutar en Colab

El notebook `notebooks/Comparacion_renal_Drive.ipynb` utiliza el backend
`shared_torch`. Está preparado para la carpeta **Proyecto renal CNN y YOLO** de
Drive, con el código y cuatro partes del ZIP original. Reconstruye el ZIP de forma
atómica y valida su SHA-256 antes de extraerlo. Monta Drive con la cuenta propietaria
de esa carpeta y selecciona un entorno GPU.

Para preparar las partes en otro equipo, descarga el ZIP de Kaggle versión 2 y usa:

```bash
python data/split_archive.py ../datos/renal.zip ../partes_drive
```

Sube esas cuatro partes y el ZIP del código a la carpeta de Drive indicada. Si la
API de Kaggle responde 404 en un entorno, usa la descarga de la versión 2 desde
su página pública y conserva el SHA-256 auditado.

Ejecuta las celdas en orden. Se guardan pruebas breves, diez entrenamientos,
evaluaciones, registros y consolidado en `resultados_t4_shared_v2`. Al repetir la
última celda se conservan las corridas completas y se reinicia la interrumpida.
La carpeta de Colab debe mantenerse separada de las corridas locales para comparar
tiempos sobre un único hardware.

El notebook anterior `Comparacion_renal_Colab.ipynb` conserva el flujo con la CNN de
Keras y YOLO de PyTorch. Es una alternativa de reproducción del backend `original`.
Las corridas definitivas de esta entrega usan el backend compartido y la GPU T4 de Colab,
registrada en cada `trained.json`.

## Fuentes principales

- [Recopilación renal de GonzaJL](https://www.kaggle.com/datasets/gonzajl/riones-cyst-stone-tumor-normal-dataset)
- [Notebook de la CNN de Ahnaf Tahmeed](https://www.kaggle.com/code/ahnaftahmeed/cyst-stone-tumor-normal-classify)
- [Clasificación con Ultralytics](https://docs.ultralytics.com/tasks/classify/)
- [Configuración YOLO26-cls en la versión 8.4.0](https://github.com/ultralytics/ultralytics/blob/v8.4.0/ultralytics/cfg/models/26/yolo26-cls.yaml)
