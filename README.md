# Documentación del servidor UDP + Flask

## Descripción general
Este servidor combina un receptor UDP para datos de sensores con una aplicación web Flask
que permite visualizar en tiempo real lecturas crudas y predicciones de actividad humana.
La información se consume mediante Server-Sent Events (SSE) y se muestra en una tabla HTML
mientras que un modelo de clasificación previamente entrenado infiere la actividad actual.

El código vive en `servidor.py` y utiliza las siguientes piezas principales:

1. **Recepción UDP** (`udp_loop`): escucha paquetes JSON provenientes de un dispositivo IoT
   (por ejemplo, un teléfono) y normaliza sus campos.
2. **Gestión de buffers** (`BUFF`, `LAST` y lógica asociada): construye una ventana móvil
   con las últimas lecturas de acelerómetro y giroscopio para alimentar al modelo.
3. **Extracción de características** (`_feat_block`, `_make_feature_vector_from_buffers`):
   calcula estadísticas para cada eje y magnitudes combinadas.
4. **Predicción con modelo MLP** (`_try_predict_and_push_row`): obtiene probabilidades y
   publica resultados en el stream.
5. **API HTTP y UI** (`Flask`): expone endpoints `/`, `/stream`, `/reset` y `/api/latest`
   para monitoreo y depuración.

## Dependencias
- **Python 3.9+** recomendado.
- Librerías estándar (`socket`, `json`, `time`, `threading`, `datetime`, `collections`).
- **Flask** para el servidor HTTP y SSE.
- **joblib** y **numpy** para cargar el modelo y procesar las características.

Los artefactos del modelo deben existir en `entreno/har_mlp.joblib` y
`entreno/har_labels.json`.

## Configuración principal
Las variables globales al inicio del archivo controlan el comportamiento del servidor:

- `UDP_IP` / `UDP_PORT`: interfaz y puerto donde llega el datastream UDP.
- `SENSORS_TO_COLLECT`: subconjunto de sensores a aceptar (`{"gyro", "accel", "gps"}`) o
  `None` para todos.
- `MAX_ROWS_MEMORY`: máximo de filas que se almacenan para mostrarse en la UI.
- `STREAM_INTERVAL_SEC`: frecuencia con la que se envían eventos SSE a los clientes.
- `HTTP_HOST` / `HTTP_PORT`: configuración del servidor Flask.
- `MODEL_PATH`, `LABELS_JSON`, `WINDOW`: rutas al modelo y etiquetas, tamaño de ventana de
  128 muestras como en el dataset UCI HAR.
- `SAMPLE_HZ`: frecuencia con la que se toma una muestra de los buffers `LAST` (60 Hz por
  defecto). De ella se deriva `SAMPLE_PERIOD`.
- `PRED_EVERY`: cantidad de muestras nuevas necesarias para volver a ejecutar la
  predicción sobre la ventana completa.

## Flujo de datos
1. **Recepción y parseo**:
   - `udp_loop` bloquea en `recvfrom` hasta recibir un paquete.
   - Se intenta decodificar a JSON y se normalizan las parejas `sensor:eje` mediante
     `iter_pairs_from_msg`.
2. **Filtrado y almacenamiento**:
   - Cada par se transforma en una fila con timestamp ISO y se agrega a `rows` (un `deque`).
   - `LAST` actualiza el último valor por eje para `accel` y `gyro`.
3. **Muestreo temporal**:
   - Cada `SAMPLE_PERIOD` segundos se toma una muestra coherente (misma marca temporal) y se
     inserta en los buffers `BUFF`.
4. **Predicción**:
   - Al acumular `WINDOW` muestras por eje y haber pasado `PRED_EVERY` muestras desde la
     última inferencia, `_try_predict_and_push_row` genera un vector de características
     concatenando estadísticas (`mean`, `std`, `min`, `max`, `rms`, `mad`).
   - Se consulta `MODEL.predict_proba` y `MODEL.predict` para obtener la etiqueta y la
     confianza máxima; se publica como una fila especial `pred:<LABEL>`.
5. **Consumo en la UI**:
   - La ruta `/stream` ofrece SSE; cada cliente solo recibe filas nuevas gracias a la
     columna `seq`.
   - El frontend actualiza la tabla con las últimas 20 filas y muestra la actividad
     inferida en el encabezado.

## Endpoints HTTP
| Ruta          | Método | Descripción |
|---------------|--------|-------------|
| `/`           | GET    | Renderiza la UI con los últimos registros y el estado actual.
| `/stream`     | GET    | Stream SSE que envía las filas nuevas en formato JSON.
| `/reset`      | GET    | Limpia buffers, reinicia contadores y marca el inicio de captura.
| `/api/latest` | GET    | Devuelve las últimas 100 filas en JSON (útil para depuración).

## Ejecución
1. Asegúrate de que `UDP_IP` sea alcanzable por el emisor (por ejemplo, la IP local de tu
   computadora en la red).
2. Lanza el script: `python servidor.py`.
3. El hilo UDP se iniciará automáticamente y Flask servirá la UI en
   `http://HTTP_HOST:HTTP_PORT`.
4. Envía paquetes UDP con estructura JSON compatible (ver formato en la siguiente sección).

## Formato esperado de mensajes UDP
El parser acepta múltiples formas:

- Campos planos `"accel:x"`, `"gyro:z"`, etc.
- Objeto `sensordata` con subdiccionarios por sensor (`{"accel": {"x": 0.1, ...}}`).
- Listas de 3 elementos (`{"gyro": [x, y, z]}`).
- Campos sueltos `x`, `y`, `z` dentro de `sensordata`, que se asignan a `gyro` o `accel`.

Cualquier valor que no pueda convertirse a `float` se ignora silenciosamente.

## Sincronización y seguridad
- Se utiliza `_start_lock` para inicializar `start_time` una única vez cuando llegan datos.
- `rows` es un `deque` compartido; las operaciones de append y lectura en Python son
  seguras para múltiples hilos en este contexto, pero podrían necesitar sincronización
  adicional si se añadieran escrituras más complejas.

## Reinicio del estado
La ruta `/reset` limpia todos los buffers y reinicia el contador `NEXT_SEQ`, útil para
comenzar una captura sin ruido residual.

## Extensiones sugeridas
- Persistir historiales en disco en lugar de depender solo de memoria.
- Exponer métricas Prometheus para monitorear tasa de paquetes y latencias.
- Añadir autenticación básica al endpoint `/reset`.
- Permitir configuración mediante variables de entorno o archivo `.env`.

