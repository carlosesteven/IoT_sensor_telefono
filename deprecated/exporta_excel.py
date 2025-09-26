import socket, json, time, os
from datetime import datetime
import pandas as pd

# ==== CONFIG ====
UDP_IP = "10.147.20.189"
UDP_PORT = 8888
DURATION_SECONDS = 10
OUT_PATH = "./sensors_long.csv"
SENSORS_TO_COLLECT = None  # o {"gyro","accel","gps"}
# ================

# Mapa de unidades por sensor
UNITS = {
    "accel": "m/s²",
    "gyro": "rad/s",
    "gps:latitude": "°",
    "gps:longitude": "°"
}

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Escuchando en {UDP_IP}:{UDP_PORT}")

def iter_pairs_from_msg(msg: dict):
    # 1) claves planas sensor:campo
    for k, v in msg.items():
        if isinstance(k, str) and ":" in k:
            sensor, field = k.split(":", 1)
            yield sensor.strip().lower() + ":" + field.strip().lower(), v

    # 2) dentro de sensordata
    sd = msg.get("sensordata")
    if isinstance(sd, dict):
        for sensor, payload in sd.items():
            s = str(sensor).lower()
            if isinstance(payload, dict):
                for field, val in payload.items():
                    yield f"{s}:{str(field).lower()}", val
            elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
                yield f"{s}:x", payload[0]
                yield f"{s}:y", payload[1]
                yield f"{s}:z", payload[2]

        # 3) ejes sueltos x,y,z
        if any(k in sd for k in ("x", "y", "z")):
            if "gyro" in sd: base = "gyro"
            elif "accel" in sd: base = "accel"
            else: base = "unknown"
            for axis in ("x", "y", "z"):
                if axis in sd:
                    yield f"{base}:{axis}", sd[axis]

rows = []
start_time = None

while True:
    data, addr = sock.recvfrom(65535)
    txt = data.decode("utf-8", errors="ignore")
    try:
        msg = json.loads(txt)
    except json.JSONDecodeError:
        continue

    pairs = list(iter_pairs_from_msg(msg))
    if not pairs:
        continue

    now = time.time()
    if start_time is None:
        start_time = now
        print(f"⏱️ Iniciando captura por {DURATION_SECONDS}s...")

    t_rel = now - start_time
    t_iso = datetime.utcnow().isoformat()

    for key, val in pairs:
        sensor_name = key.split(":", 1)[0]
        if SENSORS_TO_COLLECT and sensor_name not in SENSORS_TO_COLLECT:
            continue
        try:
            valor = float(val)
        except (TypeError, ValueError):
            continue

        # determina unidad (ej: gyro:x → gyro)
        unidad = UNITS.get(key, UNITS.get(sensor_name, ""))

        rows.append({
            "timestamp_iso": t_iso,
            "t_rel_s": t_rel,
            "sensor": key,      # ej. 'gyro:x', 'gps:latitude'
            "valor": valor,
            "unidad": unidad
        })

    if t_rel >= DURATION_SECONDS:
        df = pd.DataFrame(rows, columns=["timestamp_iso","t_rel_s","sensor","valor","unidad"])
        os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
        df.to_csv(OUT_PATH, index=False, encoding="utf-8")
        print(f"✅ Guardado {len(df)} filas en: {OUT_PATH}")
        break