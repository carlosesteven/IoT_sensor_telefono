import socket, json, time, os
from datetime import datetime
import pandas as pd

# ==== CONFIG ====
UDP_IP = "10.147.20.189"
UDP_PORT = 8888
DURATION_SECONDS = 10
OUT_PATH = "./sensors_capture.csv"
SENSORS_TO_COLLECT = {"gyro", "accel", "gps"}  # filtra lo que te interesa
# ================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Escuchando en {UDP_IP}:{UDP_PORT}")

def extract_sensors(msg: dict):
    """
    Devuelve un dict: { sensor: {campo: valor, ...}, ... }
    Acepta:
      - Claves planas: 'gyro:x', 'accel:y', 'gps:latitude', ...
      - Dentro de 'sensordata': { gyro:{x,y,z}, accel:{...}, gps:{latitude,...} }
    """
    res = {}

    # 1) claves planas sensor:campo en el root
    for k, v in msg.items():
        if isinstance(k, str) and ":" in k:
            sensor, field = k.split(":", 1)
            sensor = sensor.strip().lower()
            field = field.strip().lower()
            res.setdefault(sensor, {})[field] = v

    # 2) dentro de 'sensordata'
    sd = msg.get("sensordata")
    if isinstance(sd, dict):
        for sensor, payload in sd.items():
            s = str(sensor).lower()
            if isinstance(payload, dict):
                for field, val in payload.items():
                    res.setdefault(s, {})[str(field).lower()] = val
            elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
                # lista de ejes [x,y,z]
                res.setdefault(s, {}).update({"x": payload[0], "y": payload[1], "z": payload[2]})
        # Caso “mixto”: ejes sueltos x,y,z al mismo nivel → asócialos a 'gyro' si no hay otro mejor
        if any(k in sd for k in ("x","y","z")):
            target = "gyro" if "gyro" in sd else "accel" if "accel" in sd else "unknown"
            res.setdefault(target, {})
            for axis in ("x","y","z"):
                if axis in sd:
                    res[target][axis] = sd[axis]

    return res

# --- captura y guardado ---
rows = []
start_time = None

while True:
    data, addr = sock.recvfrom(65535)
    txt = data.decode("utf-8", errors="ignore")
    try:
        msg = json.loads(txt)
    except json.JSONDecodeError:
        continue

    sensors = extract_sensors(msg)
    if not sensors:
        # No hay nada que mapear
        continue

    now = time.time()
    if start_time is None:
        start_time = now
        print(f"⏱️ Iniciando captura por {DURATION_SECONDS}s...")

    t_rel = now - start_time
    t_iso = datetime.utcnow().isoformat()

    for sensor, fields in sensors.items():
        sensor = sensor.lower()
        if SENSORS_TO_COLLECT and sensor not in SENSORS_TO_COLLECT:
            continue

        # Normaliza campos
        x = fields.get("x")
        y = fields.get("y")
        z = fields.get("z")
        lat = fields.get("latitude") or fields.get("lat")
        lon = fields.get("longitude") or fields.get("lon")

        # imprime algo útil
        if sensor in ("gyro", "accel"):
            if None not in (x, y, z):
                print(f"{sensor}: x={float(x):.6f} y={float(y):.6f} z={float(z):.6f}")
        elif sensor == "gps":
            if lat is not None and lon is not None:
                print(f"gps: lat={float(lat):.6f} lon={float(lon):.6f}")

        # agrega fila; columnas comunes (deja NaN donde no aplique)
        rows.append({
            "timestamp_iso": t_iso,
            "t_rel_s": t_rel,
            "sensor": sensor,
            "x": float(x) if x is not None else None,
            "y": float(y) if y is not None else None,
            "z": float(z) if z is not None else None,
            "latitude": float(lat) if lat is not None else None,
            "longitude": float(lon) if lon is not None else None,
        })

    if t_rel >= DURATION_SECONDS:
        df = pd.DataFrame(rows, columns=[
            "timestamp_iso","t_rel_s","sensor","x","y","z","latitude","longitude"
        ])
        os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
        df.to_csv(OUT_PATH, index=False, encoding="utf-8")
        print(f"Guardado {len(df)} filas en: {OUT_PATH}")
        break