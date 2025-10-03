#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
import json
import time
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template


# ==================== CONFIG ====================
UDP_HOST = "0.0.0.0"      # escucha en todas las interfaces
UDP_PORT = 8080           # puerto donde Serial Sensor envía
WINDOW_SEC = 1.0          # duración de ventana en segundos
STEP_SEC = 0.5            # cada cuánto predecir
MODEL_PATH = "entrenamiento/IA_Movimientos.joblib"  # ruta fija a tu modelo

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 5050
POLL_MS   = 500           # refresco del frontend

# ==================== FEATURES ====================
def ingenieriaCaracteristicas(df: pd.DataFrame):
    """Genera cinco descriptores numéricos a partir de una ventana de aceleraciones."""
    # Calcula la media del eje X como medida del componente longitudinal.
    F1 = df['Ax'].mean()
    # Calcula la media del eje Y para identificar tendencias laterales.
    F2 = df['Ay'].mean()
    # Calcula la media del eje Z para capturar el efecto vertical.
    F3 = df['Az'].mean()
    # Calcula la media de las desviaciones estándar para sintetizar la variabilidad total.
    F4 = (df['Ax'].std() + df['Ay'].std() + df['Az'].std()) / 3.0
    # Calcula la magnitud promedio del vector de aceleración como energía del movimiento.
    F5 = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2).mean()
    # Devuelve las características convertidas explícitamente a float.
    return [float(F1), float(F2), float(F3), float(F4), float(F5)]

# ==================== DECODIFICACIÓN ====================
MESSAGE_LENGTH = 13
SENSOR_TAG_MAP = {ord('A'): "acc", ord('a'): "acc", 1: "acc"}

def decode_serialsensor_binary(pkt: bytes):
    """Lee un paquete binario de 13 bytes y extrae aceleraciones en formato flotante."""
    # Verifica que el mensaje tenga exactamente el tamaño esperado.
    if len(pkt) != MESSAGE_LENGTH:
        return None
    # Recupera el byte de encabezado que identifica el tipo de sensor.
    tag = pkt[0]
    # Busca el prefijo textual asociado al identificador.
    pref = SENSOR_TAG_MAP.get(tag)
    # Descarta el paquete si no corresponde a un acelerómetro.
    if pref != "acc":
        return None
    try:
        # Desempaqueta el valor del eje X usando formato float little-endian.
        x = struct.unpack_from('<f', pkt, 1)[0]
        # Desempaqueta el eje Y a partir del offset 5.
        y = struct.unpack_from('<f', pkt, 5)[0]
        # Desempaqueta el eje Z desde el offset 9.
        z = struct.unpack_from('<f', pkt, 9)[0]
        # Devuelve los tres ejes normalizados como flotantes.
        return ("acc", float(x), float(y), float(z))
    except Exception:
        # En caso de error de desempaquetado se descarta el paquete.
        return None

def iter_pairs_from_json_msg(msg: dict):
    """Extrae triples (sensor, eje, valor) de las variantes JSON aceptadas."""
    # Recorre todos los pares clave/valor del mensaje original.
    for k, v in list(msg.items()):
        # Solo procesa claves que contengan el separador sensor:eje.
        if isinstance(k, str) and ":" in k:
            # Divide en prefijo y eje usando la primera aparición de ":".
            s, a = k.split(":", 1)
            # Normaliza ambos fragmentos a minúsculas sin espacios externos.
            s, a = s.strip().lower(), a.strip().lower()
            # Comprueba que el prefijo sea de acelerómetro y que el eje sea válido.
            if s in ("accel","accelerometer","acc") and a in ("x","y","z"):
                # Produce el par normalizado para su consumo en el pipeline.
                yield "acc", a, v
    # Examina si existe una sección "sensordata" más estructurada.
    sd = msg.get("sensordata")
    # Continúa solo si dicha sección es un diccionario.
    if isinstance(sd, dict):
        # Itera sobre cada sensor descrito dentro del bloque.
        for sensor, payload in sd.items():
            # Verifica que el nombre del sensor corresponda a un acelerómetro.
            if str(sensor).lower().startswith("acc"):
                if isinstance(payload, dict):
                    # Recorre los ejes explícitos del diccionario interno.
                    for axis, val in payload.items():
                        # Normaliza el nombre del eje para validarlo.
                        a = str(axis).lower()
                        if a in ("x","y","z"):
                            # Devuelve cada eje acompañado de su valor.
                            yield "acc", a, val
                elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
                    # Para secuencias, asigna los primeros tres valores a X, Y, Z respectivamente.
                    for a, val in zip(("x","y","z"), payload[:3]):
                        yield "acc", a, val

# ==================== ESTADO COMPARTIDO ====================
state_lock = threading.Lock()
window = deque()  # (t, Ax, Ay, Az)
last_pred = {
    "label": None,            # "R" o "L"
    "prob_rapido": None,      # float 0..1
    "F": [None]*5,            # F1..F5
    "n": 0,                   # muestras en ventana
    "updated_at": None,       # timestamp texto
}

# ==================== MODELO ====================
model = None
f5_threshold_default = 13.1425
try:
    model = joblib.load(MODEL_PATH)
    print(f"[MODEL] Modelo cargado: {MODEL_PATH}")
except Exception as e:
    print(f"[MODEL] No se pudo cargar {MODEL_PATH}: {e}. Usaré umbral F5 por defecto {f5_threshold_default}.")

# ==================== HILO UDP + PREDICCIÓN ====================
def udp_loop():
    """Escucha paquetes UDP, mantiene la ventana y actualiza la predicción compartida."""
    # Crea un socket UDP para recibir lecturas del sensor.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Vincula el socket a la interfaz y puerto configurados.
    sock.bind((UDP_HOST, UDP_PORT))
    # Define un timeout para evitar bloqueos indefinidos.
    sock.settimeout(1.0)
    # Informa en consola que el listener está activo.
    print(f"[UDP] Escuchando en {UDP_HOST}:{UDP_PORT}")

    # Registra el instante de la última predicción para respetar STEP_SEC.
    last_pred_t = 0.0
    try:
        while True:
            # Obtiene el tiempo actual al inicio del ciclo.
            now = time.time()
            # Calcula el límite inferior permitido en la ventana deslizante.
            cutoff = now - WINDOW_SEC
            with state_lock:
                # Elimina muestras antiguas que queden fuera de la ventana.
                while window and window[0][0] < cutoff:
                    window.popleft()

            try:
                # Intenta recibir un datagrama desde el socket.
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                # Si no llegan datos a tiempo, sigue el ciclo.
                data = None

            if data:
                # Maneja paquetes binarios del tamaño esperado para Serial Sensor.
                if len(data) == MESSAGE_LENGTH:
                    decoded = decode_serialsensor_binary(data)
                    if decoded:
                        # Obtiene las componentes XYZ y las incorpora a la ventana.
                        _, x, y, z = decoded
                        with state_lock:
                            window.append((now, x, y, z))
                # Maneja mensajes JSON cuando el primer byte indica "{" o "[".
                elif data[:1] in (b'{', b'['):
                    try:
                        # Decodifica el paquete a texto y luego a objeto Python.
                        msg = json.loads(data.decode("utf-8", errors="ignore"))
                    except Exception:
                        # Descarta el mensaje si la decodificación falla.
                        msg = None
                    if isinstance(msg, dict):
                        # Inicializa el diccionario para asegurarse de completar los tres ejes.
                        axes = {"x": None, "y": None, "z": None}
                        for pref, axis, val in iter_pairs_from_json_msg(msg):
                            if pref == "acc":
                                try:
                                    # Intenta convertir el valor recibido a float.
                                    axes[axis] = float(val)
                                except (TypeError, ValueError):
                                    # Ignora valores no numéricos manteniendo el eje como None.
                                    pass
                        if None not in axes.values():
                            with state_lock:
                                # Inserta la medición completa en la ventana.
                                window.append((now, axes["x"], axes["y"], axes["z"]))

            # Comprueba si ya transcurrió el paso mínimo para evaluar el modelo.
            if (now - last_pred_t) >= STEP_SEC:
                # Actualiza el marcador de tiempo de la última predicción.
                last_pred_t = now
                with state_lock:
                    # Calcula el número de muestras disponibles en la ventana.
                    n = len(window)
                    if n >= 3:
                        # Transforma la ventana a arreglo NumPy con formato consistente.
                        arr = np.array([[t, ax, ay, az] for (t, ax, ay, az) in window], dtype=float)
                        # Construye un DataFrame con los ejes para reutilizar la ingeniería de características.
                        df_win = pd.DataFrame(arr[:, 1:], columns=["Ax","Ay","Az"])
                        # Calcula las características derivadas de la ventana actual.
                        F = ingenieriaCaracteristicas(df_win)

                        if model is not None:
                            try:
                                # Obtiene la probabilidad de la clase "R" desde el modelo entrenado.
                                prob = float(model.predict_proba([F])[0, 1])
                                # Define la etiqueta en función del umbral 0.5 habitual.
                                label = "R" if prob >= 0.5 else "L"
                            except Exception as e:
                                # Si el modelo falla en inferencia, recurre al plan de contingencia.
                                prob = 1.0 / (1.0 + np.exp(-(F[4] - f5_threshold_default)))
                                label = "R" if F[4] > f5_threshold_default else "L"
                        else:
                            # Sin modelo cargado, utiliza la sigmoide sobre F5 como aproximación.
                            prob = 1.0 / (1.0 + np.exp(-(F[4] - f5_threshold_default)))
                            label = "R" if F[4] > f5_threshold_default else "L"

                        # Actualiza el estado compartido con la predicción y metadatos auxiliares.
                        last_pred.update({
                            "label": label,
                            "prob_rapido": prob,
                            "F": [round(x, 6) for x in F],
                            "n": n,
                            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
    finally:
        try:
            # Cierra el socket al salir del bucle principal.
            sock.close()
        except Exception:
            # Ignora cualquier error al cerrar para no interrumpir el apagado.
            pass

# ==================== APP WEB (Flask) ====================
app = Flask(__name__)

@app.route("/")
def index():
    """Renderiza la plantilla del dashboard pasando los parámetros de configuración."""
    # Devuelve la página HTML principal con los datos de contexto necesarios para la UI.
    return render_template(
        "dashboard.html",
        win=WINDOW_SEC,
        step=STEP_SEC,
        udp_host=UDP_HOST,
        udp_port=UDP_PORT,
        model_path=MODEL_PATH,
        poll=POLL_MS
    )

@app.route("/api/status")
def api_status():
    """Entrega el último estado de predicción en formato JSON para el frontend."""
    # Protege el acceso al diccionario compartido con el candado global.
    with state_lock:
        payload = dict(last_pred)
    # Serializa la información del estado y la retorna como respuesta HTTP.
    return jsonify(payload)

# ==================== ENTRYPOINT ====================
if __name__ == "__main__":
    # Lanzar el hilo del listener UDP
    t = threading.Thread(target=udp_loop, daemon=True)
    t.start()
    # Levantar servidor web
    print(f"[WEB] http://{HTTP_HOST}:{HTTP_PORT}")
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False)
