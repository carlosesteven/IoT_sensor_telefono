#!/usr/bin/env python3
import socket
import struct
import json
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

# ==================== CONFIG ====================
UDP_HOST = "0.0.0.0"   # escucha en todas las interfaces
UDP_PORT = 8080        # puerto donde Serial Sensor envía
WINDOW_SEC = 1.0       # duración de ventana en segundos
STEP_SEC = 0.5         # cada cuánto predecir
MODEL_PATH = "entrenamiento/IA_Movimientos.joblib"  # ruta fija a tu modelo

# ==================== FEATURES ====================
def ingenieriaCaracteristicas(df: pd.DataFrame):
    """Calcula cinco características estadísticas a partir de una ventana de aceleraciones."""
    # Calcula la media del eje X para estimar la inclinación o sesgo en esa dirección.
    F1 = df['Ax'].mean()
    # Calcula la media del eje Y para detectar tendencia de aceleración lateral.
    F2 = df['Ay'].mean()
    # Calcula la media del eje Z para captar el componente vertical del movimiento.
    F3 = df['Az'].mean()
    # Promedia las desviaciones estándar de los tres ejes como indicador de variabilidad global.
    F4 = (df['Ax'].std() + df['Ay'].std() + df['Az'].std()) / 3.0
    # Obtiene la magnitud media del vector de aceleración (norma) como energía del movimiento.
    F5 = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2).mean()
    # Devuelve la lista con las cinco características en el orden esperado por el modelo.
    return [F1, F2, F3, F4, F5]

# ==================== DECODIFICACIÓN ====================
MESSAGE_LENGTH = 13
SENSOR_TAG_MAP = {ord('A'): "acc", ord('a'): "acc", 1: "acc"}

def decode_serialsensor_binary(pkt: bytes):
    """Interpreta un paquete binario de Serial Sensor y extrae aceleraciones XYZ."""
    # Valida que el paquete tenga exactamente la longitud esperada de 13 bytes.
    if len(pkt) != MESSAGE_LENGTH:
        return None
    # Obtiene el primer byte, que representa el identificador del sensor.
    tag = pkt[0]
    # Traduce el identificador a un prefijo textual según el mapa permitido.
    pref = SENSOR_TAG_MAP.get(tag)
    # Si el prefijo no corresponde a acelerómetro, se descarta el paquete.
    if pref != "acc":
        return None
    try:
        # Extrae el valor de aceleración X interpretando 4 bytes a partir del offset 1.
        x = struct.unpack_from('<f', pkt, 1)[0]
        # Extrae el valor Y interpretando otros 4 bytes desde el offset 5.
        y = struct.unpack_from('<f', pkt, 5)[0]
        # Extrae el valor Z con 4 bytes adicionales a partir del offset 9.
        z = struct.unpack_from('<f', pkt, 9)[0]
        # Devuelve los tres ejes en formato de tupla normalizada.
        return ("acc", float(x), float(y), float(z))
    except Exception:
        # Cualquier error de desempaquetado se captura y se devuelve None.
        return None

def iter_pairs_from_json_msg(msg: dict):
    """Normaliza distintas variantes de mensajes JSON y produce pares (sensor, eje, valor)."""
    # Recorre una copia de los pares clave/valor para poder modificar sin afectar el original.
    for k, v in list(msg.items()):
        # Solo procesa claves que contengan separador de sensor:eje.
        if isinstance(k, str) and ":" in k:
            # Divide la clave en prefijo de sensor y eje individual.
            s, a = k.split(":", 1)
            # Limpia espacios y normaliza a minúsculas.
            s, a = s.strip().lower(), a.strip().lower()
            # Comprueba que el prefijo represente un acelerómetro y que el eje sea válido.
            if s in ("accel","accelerometer","acc") and a in ("x","y","z"):
                # Entrega el eje correspondiente junto con su valor original.
                yield "acc", a, v
    # Obtiene el subobjeto "sensordata" si está presente.
    sd = msg.get("sensordata")
    # Verifica que el subobjeto sea un diccionario para inspeccionarlo.
    if isinstance(sd, dict):
        # Itera por cada sensor incluido dentro de la clave "sensordata".
        for sensor, payload in sd.items():
            # Filtra aquellos cuyo nombre indique acelerómetro.
            if str(sensor).lower().startswith("acc"):
                # Si el payload es una secuencia de tres valores, los asigna a ejes XYZ.
                if isinstance(payload, (list, tuple)) and len(payload) >= 3:
                    # Empareja cada eje con el valor correspondiente de la secuencia.
                    for a, val in zip(("x","y","z"), payload[:3]):
                        yield "acc", a, val

# ==================== MAIN ====================
def main():
    """Gestiona el ciclo principal: carga el modelo, escucha UDP y emite predicciones."""
    # Intenta cargar el modelo entrenado desde disco para obtener probabilidades reales.
    try:
        model = joblib.load(MODEL_PATH)
        # Confirma por consola que el modelo se cargó correctamente.
        print(f"[MODEL] Modelo cargado: {MODEL_PATH}")
    except Exception as e:
        # Si falla la carga, se recurre a un umbral manual sobre la característica F5.
        model = None
        print(f"[MODEL] No se pudo cargar {MODEL_PATH}: {e}")
        # Define el umbral empírico que separa movimientos rápidos de lentos.
        f5_threshold = 13.1425

    # Crea el socket UDP que recibirá las lecturas de aceleración.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Asocia el socket a la interfaz y puerto configurados.
    sock.bind((UDP_HOST, UDP_PORT))
    # Establece un timeout para evitar bloqueo indefinido en recvfrom.
    sock.settimeout(1.0)
    # Informa dónde está escuchando el servidor.
    print(f"[UDP] Escuchando en {UDP_HOST}:{UDP_PORT}")

    # Crea la ventana deslizante que acumula muestras recientes.
    window = deque()
    # Guarda el instante de la última predicción para espaciar las ejecuciones.
    last_pred_t = 0.0

    try:
        while True:
            # Registra el tiempo actual para purgar la ventana y etiquetar muestras.
            now = time.time()
            # Determina el límite inferior de la ventana según WINDOW_SEC.
            cutoff = now - WINDOW_SEC
            # Elimina muestras más antiguas que el corte temporal.
            while window and window[0][0] < cutoff:
                window.popleft()

            try:
                # Intenta recibir un datagrama UDP de hasta 65535 bytes.
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                # Si se agota el tiempo sin datos, continúa el ciclo sin fallar.
                data = None

            if data:
                # Caso de trama binaria exacta proveniente de Serial Sensor.
                if len(data) == MESSAGE_LENGTH:
                    decoded = decode_serialsensor_binary(data)
                    if decoded:
                        # Extrae los tres ejes y los añade a la ventana con el timestamp actual.
                        _, x, y, z = decoded
                        window.append((now, x, y, z))
                # Caso de mensaje JSON (objeto o lista) identificado por llaves o corchetes.
                elif data[:1] in (b'{', b'['):
                    try:
                        # Decodifica el payload a texto UTF-8 y luego a estructura Python.
                        msg = json.loads(data.decode("utf-8", errors="ignore"))
                    except:
                        # Si falla la decodificación, se descarta el paquete.
                        msg = None
                    if isinstance(msg, dict):
                        # Inicializa un contenedor para asegurar que existan los tres ejes.
                        axes = {"x": None, "y": None, "z": None}
                        # Itera sobre las variantes admitidas y rellena los ejes encontrados.
                        for pref, axis, val in iter_pairs_from_json_msg(msg):
                            if pref == "acc":
                                axes[axis] = float(val)
                        # Solo agrega a la ventana si están presentes los tres ejes.
                        if None not in axes.values():
                            window.append((now, axes["x"], axes["y"], axes["z"]))

            # Comprueba si ya pasó el intervalo mínimo y hay datos suficientes para predecir.
            if (now - last_pred_t) >= STEP_SEC and len(window) > 3:
                # Actualiza el instante de la última inferencia realizada.
                last_pred_t = now
                # Convierte la ventana a un array NumPy para manipularla eficientemente.
                arr = np.array([[t, ax, ay, az] for (t, ax, ay, az) in window])
                # Crea un DataFrame solo con columnas de aceleración para las features.
                df_win = pd.DataFrame(arr[:,1:], columns=["Ax","Ay","Az"])
                # Calcula las cinco características necesarias para el clasificador.
                F = ingenieriaCaracteristicas(df_win)

                if model:
                    # Obtiene la probabilidad de movimiento rápido (clase positiva).
                    prob = model.predict_proba([F])[0,1]
                    # Asigna la etiqueta según el umbral 0.5 típico de modelos binarios.
                    pred = "R" if prob >= 0.5 else "L"
                else:
                    # En modo sin modelo, compara la energía (F5) con el umbral definido.
                    pred = "R" if F[4] > f5_threshold else "L"
                    # Aproxima una probabilidad suave aplicando una sigmoide alrededor del umbral.
                    prob = 1 / (1 + np.exp(-(F[4]-f5_threshold)))

                # Genera una marca de tiempo legible para la consola.
                ts = datetime.now().strftime("%H:%M:%S")
                # Imprime la predicción, probabilidad y valores de las características.
                print(f"[{ts}] Pred={pred} Prob_Rapido={prob:.3f} "
                      f"F1={F[0]:.2f} F2={F[1]:.2f} F3={F[2]:.2f} F4={F[3]:.2f} F5={F[4]:.2f} n={len(window)}")

    except KeyboardInterrupt:
        # Permite terminar el programa con Ctrl+C mostrando un mensaje amistoso.
        print("\n[EXIT] Interrumpido por usuario.")
    finally:
        # Garantiza que el socket se cierre aunque ocurra una excepción.
        sock.close()

if __name__ == "__main__":
    main()
