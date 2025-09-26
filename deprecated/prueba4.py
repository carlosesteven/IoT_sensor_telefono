import socket, json, time, os
from datetime import datetime
import pandas as pd

# =========================
# CONFIGURACIÓN
# =========================
UDP_IP = "10.147.20.189"
UDP_PORT = 8888

# Cuánto tiempo guardar desde que llega el primer paquete válido (en segundos)
DURATION_SECONDS = 10

# Si quieres filtrar por sensores específicos (p.ej. ["gyro", "accel"])
# deja en None para guardar todo lo que venga en "sensordata"
SENSORS_TO_COLLECT = ["gyro"]  # o None

# Carpeta y nombre de salida
OUT_DIR = "."
FILE_PREFIX = "sensors"
# =========================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(1.0)  # evita quedarse bloqueado para siempre

print(f"Escuchando en {UDP_IP}:{UDP_PORT}")

rows = []  # juntamos aquí y luego volcamos a pandas
start_time = None

def merge_axes_from(sd: dict, sensor: str):
    """
    Devuelve un dict con posibles ejes x,y,z tomando:
      1) ejes planos en sd (caso mixto): sd['x'], sd['y'], sd['z']
      2) ejes dentro de sd[sensor] si es dict o lista
      3) claves planas 'sensor:x' ... en el root de 'msg' (no se usa aquí)
    """
    out = {"x": None, "y": None, "z": None}

    # 1) planos
    for k in ("x", "y", "z"):
        if k in sd:
            out[k] = sd[k]

    # 2) dentro de sd[sensor]
    g = sd.get(sensor)
    if isinstance(g, dict):
        for k in ("x", "y", "z"):
            if g.get(k) is not None:
                out[k] = g[k]
    elif isinstance(g, (list, tuple)) and len(g) >= 3:
        out["x"], out["y"], out["z"] = g[0], g[1], g[2]

    return out

def extract_sensor_rows(msg: dict):
    """
    Convierte un mensaje en filas (formato largo) tipo:
      {timestamp_iso, t_rel_s, device, uuid, sensor, x, y, z}
    """
    global start_time
    now = time.time()
    if start_time is None:
        start_time = now

    t_rel = now - start_time
    t_iso = datetime.utcnow().isoformat()

    device = (msg.get("device") or {}).get("name")
    uuid = msg.get("uuid")
    sd = msg.get("sensordata") or {}

    rows_local = []

    # Determinar qué sensores tomar
    sensor_names = []
    if SENSORS_TO_COLLECT is None:
        # todos los que estén dentro de sensordata
        # (incluir también ejes planos x,y,z “huérfanos” bajo un sensor “unknown”)
        sensor_names = [k for k in sd.keys() if isinstance(sd.get(k), (dict, list, tuple))]
        # si hay ejes sueltos y no hay un sensor declarado, asignarlos a 'unknown'
        if any(k in sd for k in ("x", "y", "z")) and "unknown" not in sensor_names:
            sensor_names.append("unknown")
    else:
        sensor_names = SENSORS_TO_COLLECT

    for sensor in sensor_names:
        if sensor == "unknown":
            axes = {"x": sd.get("x"), "y": sd.get("y"), "z": sd.get("z")}
        else:
            axes = merge_axes_from(sd, sensor)

        # Si no hay ningún valor, omite
        if all(v is None for v in axes.values()):
            continue

        try:
            rows_local.append({
                "timestamp_iso": t_iso,
                "t_rel_s": t_rel,
                "device": device,
                "uuid": uuid,
                "sensor": sensor,
                "x": float(axes["x"]) if axes["x"] is not None else None,
                "y": float(axes["y"]) if axes["y"] is not None else None,
                "z": float(axes["z"]) if axes["z"] is not None else None,
            })
        except (TypeError, ValueError):
            # Algún valor no convertible a float → lo ignoramos
            pass

    return rows_local

# Bucle de captura hasta que pase DURATION_SECONDS
while True:
    if start_time and (time.time() - start_time >= DURATION_SECONDS):
        break

    try:
        data, addr = sock.recvfrom(65535)
    except socket.timeout:
        continue

    txt = data.decode("utf-8", errors="ignore")
    try:
        msg = json.loads(txt)
    except json.JSONDecodeError:
        continue

    rows.extend(extract_sensor_rows(msg))

# Guardar con pandas
if not rows:
    print("No se capturaron datos.")
else:
    df = pd.DataFrame(rows)
    # Ordena columnas para que sea consistente
    df = df[["timestamp_iso", "t_rel_s", "device", "uuid", "sensor", "x", "y", "z"]]

    # Nombre de archivo con timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{FILE_PREFIX}_{ts}.csv"
    out_path = os.path.join(OUT_DIR, fname)

    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Guardado {len(df)} filas en: {out_path}")