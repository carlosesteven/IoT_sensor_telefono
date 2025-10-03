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
    F1 = df['Ax'].mean()
    F2 = df['Ay'].mean()
    F3 = df['Az'].mean()
    F4 = (df['Ax'].std() + df['Ay'].std() + df['Az'].std()) / 3.0
    F5 = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2).mean()
    return [F1, F2, F3, F4, F5]

# ==================== DECODIFICACIÓN ====================
MESSAGE_LENGTH = 13
SENSOR_TAG_MAP = {ord('A'): "acc", ord('a'): "acc", 1: "acc"}

def decode_serialsensor_binary(pkt: bytes):
    if len(pkt) != MESSAGE_LENGTH:
        return None
    tag = pkt[0]
    pref = SENSOR_TAG_MAP.get(tag)
    if pref != "acc":
        return None
    try:
        x = struct.unpack_from('<f', pkt, 1)[0]
        y = struct.unpack_from('<f', pkt, 5)[0]
        z = struct.unpack_from('<f', pkt, 9)[0]
        return ("acc", float(x), float(y), float(z))
    except Exception:
        return None

def iter_pairs_from_json_msg(msg: dict):
    for k, v in list(msg.items()):
        if isinstance(k, str) and ":" in k:
            s, a = k.split(":", 1)
            s, a = s.strip().lower(), a.strip().lower()
            if s in ("accel","accelerometer","acc") and a in ("x","y","z"):
                yield "acc", a, v
    sd = msg.get("sensordata")
    if isinstance(sd, dict):
        for sensor, payload in sd.items():
            if str(sensor).lower().startswith("acc"):
                if isinstance(payload, (list, tuple)) and len(payload) >= 3:
                    for a, val in zip(("x","y","z"), payload[:3]):
                        yield "acc", a, val

# ==================== MAIN ====================
def main():
    # Cargar modelo entrenado
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[MODEL] Modelo cargado: {MODEL_PATH}")
    except Exception as e:
        model = None
        print(f"[MODEL] No se pudo cargar {MODEL_PATH}: {e}")
        f5_threshold = 13.1425

    # Socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(1.0)
    print(f"[UDP] Escuchando en {UDP_HOST}:{UDP_PORT}")

    window = deque()
    last_pred_t = 0.0

    try:
        while True:
            now = time.time()
            cutoff = now - WINDOW_SEC
            while window and window[0][0] < cutoff:
                window.popleft()

            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                data = None

            if data:
                if len(data) == MESSAGE_LENGTH:
                    decoded = decode_serialsensor_binary(data)
                    if decoded:
                        _, x, y, z = decoded
                        window.append((now, x, y, z))
                elif data[:1] in (b'{', b'['):
                    try:
                        msg = json.loads(data.decode("utf-8", errors="ignore"))
                    except:
                        msg = None
                    if isinstance(msg, dict):
                        axes = {"x": None, "y": None, "z": None}
                        for pref, axis, val in iter_pairs_from_json_msg(msg):
                            if pref == "acc":
                                axes[axis] = float(val)
                        if None not in axes.values():
                            window.append((now, axes["x"], axes["y"], axes["z"]))

            if (now - last_pred_t) >= STEP_SEC and len(window) > 3:
                last_pred_t = now
                arr = np.array([[t, ax, ay, az] for (t, ax, ay, az) in window])
                df_win = pd.DataFrame(arr[:,1:], columns=["Ax","Ay","Az"])
                F = ingenieriaCaracteristicas(df_win)

                if model:
                    prob = model.predict_proba([F])[0,1]
                    pred = "R" if prob >= 0.5 else "L"
                else:
                    pred = "R" if F[4] > f5_threshold else "L"
                    prob = 1 / (1 + np.exp(-(F[4]-f5_threshold)))

                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] Pred={pred} Prob_Rapido={prob:.3f} "
                      f"F1={F[0]:.2f} F2={F[1]:.2f} F3={F[2]:.2f} F4={F[3]:.2f} F5={F[4]:.2f} n={len(window)}")

    except KeyboardInterrupt:
        print("\n[EXIT] Interrumpido por usuario.")
    finally:
        sock.close()

if __name__ == "__main__":
    main()
