# Servidores de inferencia desde sensores UDP

Este repositorio contiene dos implementaciones para capturar lecturas de acelerómetro
provenientes de Serial Sensor (o fuentes compatibles) vía UDP, extraer características y
clasificar el movimiento como **Rápido (R)** o **Lento (L)** con un modelo entrenado
previamente.

* `servidor_basico.py`: versión de consola que imprime las predicciones en tiempo real.
* `servidor_web.py`: variante con tablero web (Flask) que expone el estado mediante una API
  y una interfaz HTML.

## Requisitos previos

- Python **3.9 o superior**.
- Un archivo de modelo `IA_Movimientos.joblib` entrenado previamente (ubicado según cada
  script).
- Paquetes UDP emitidos por el dispositivo IoT con formato binario de Serial Sensor o JSON
  con campos `accel`/`acc` en ejes `x`, `y`, `z`.

## Preparar el entorno

Se recomienda aislar las dependencias en un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Instalación de librerías

Ambos scripts comparten dependencias científicas para el cálculo de características y la
carga del modelo. Además, la versión web necesita Flask para la API.

```bash
pip install numpy pandas joblib
pip install flask  # requerido solo para servidor_web.py
```

Si prefieres un único comando, puedes instalar todo de una vez:

```bash
pip install numpy pandas joblib flask
```

## Variables principales

Cada script declara constantes de configuración al inicio del archivo:

- `UDP_HOST` y `UDP_PORT`: interfaz y puerto donde se reciben las lecturas.
- `WINDOW_SEC`: segundos considerados en la ventana deslizante para generar características.
- `STEP_SEC`: intervalo mínimo entre predicciones consecutivas.
- `MODEL_PATH`: ruta relativa al artefacto `.joblib` con el modelo de clasificación.
- `HTTP_HOST`, `HTTP_PORT` y `POLL_MS`: exclusivos de `servidor_web.py` para el servicio
  Flask.

## Ejecución del servidor básico

1. Copia o enlaza tu archivo `IA_Movimientos.joblib` en `entrenamiento/`.
2. Activa el entorno virtual y ve al directorio del proyecto.
3. Lanza el script:

   ```bash
   python servidor_basico.py
   ```

4. El programa escuchará en `UDP_HOST:UDP_PORT`, mantendrá una ventana deslizante y mostrará
   la predicción junto a las cinco características (F1–F5) en consola.

## Ejecución del servidor web

1. Asegúrate de colocar `IA_Movimientos.joblib` en la raíz del repositorio (o ajusta la
   constante `MODEL_PATH`).
2. Arranca el proceso:

   ```bash
   python servidor_web.py
   ```

3. Se inicia un hilo dedicado a escuchar UDP y a calcular las predicciones. El hilo va
   actualizando un estado compartido que el frontend consulta periódicamente.
4. Accede a `http://HTTP_HOST:HTTP_PORT` para visualizar el tablero. El endpoint
   `/api/status` devuelve el último resultado en JSON, útil para integraciones.

### Ejecutar `servidor_web.py` con `nohup`

Para mantener el servidor web activo después de cerrar la terminal puedes usar `nohup`.

- **Sin registro alguno** (descarta toda la salida estándar y de error):

  ```bash
  nohup python servidor_web.py > /dev/null 2>&1 &
  ```

- **Sin archivo de log dedicado** (la salida se almacena en `nohup.out` por defecto):

  ```bash
  nohup python servidor_web.py &
  ```

- **Con un archivo de log específico** para revisar la salida estándar y de error:

  ```bash
  nohup python servidor_web.py > servidor_web.log 2>&1 &
  ```

## Formatos de mensaje admitidos

Ambos servidores aceptan:

- Tramas binarias de 13 bytes con la firma de Serial Sensor (`A` + tres flotantes little
  endian).
- Mensajes JSON con claves tipo `"accel:x"`, o estructuras anidadas bajo `sensordata`
  (`{"accel": {"x": ..., "y": ..., "z": ...}}`). Los valores deben ser numéricos.

Los datos se filtran para asegurar que existan los tres ejes antes de alimentar la ventana.

## Salida de predicción

Los cinco atributos calculados son:

1. **F1** – media de `Ax`.
2. **F2** – media de `Ay`.
3. **F3** – media de `Az`.
4. **F4** – media de las desviaciones estándar de los tres ejes.
5. **F5** – magnitud promedio del vector de aceleración.

Cuando el modelo no puede cargarse, ambos scripts aplican un umbral empírico sobre **F5**
para distinguir entre movimientos rápidos y lentos.

## Depuración y personalización

- Ajusta `WINDOW_SEC` y `STEP_SEC` según la cadencia de paquetes que emite tu dispositivo.
- Si utilizas otro formato de mensaje, amplía las funciones de decodificación en cada
  archivo (sección "DECODIFICACIÓN").
- Puedes sustituir el modelo por otro artefacto compatible con `scikit-learn` siempre que
  exponga `predict_proba`.

## Apagar los servidores

Presiona `Ctrl+C` en la terminal para detener cualquiera de los scripts. Los sockets se
cierran limpiamente en ambos casos.
