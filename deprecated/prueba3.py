import socket, json, time, os
from datetime import datetime
import pandas as pd  # ← pandas

# ==== CONFIG ====
UDP_IP = "10.147.20.189"
UDP_PORT = 8888
DURATION_SECONDS = 10            # segundos a guardar desde el primer paquete válido
OUT_PATH = "./gyro_capture.csv"  # ruta del CSV de salida
# ================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print(f"Escuchando en {UDP_IP}:{UDP_PORT}")

def get_gyro(msg):
    """
    Maneja estos casos:
      sensordata: { gyro: {x,y,z}, x, y, z }  # mixto (tu caso)
      sensordata: { x,y,z }                   # sin 'gyro'
      sensordata: { gyro: [x,y,z] }           # lista
      { 'gyro:x':.., 'gyro:y':.., 'gyro:z':.. }  # plano
    """
    sd = msg.get("sensordata") or {}

    # Prioridad: tomar x,y,z en sd si existen; si falta alguno, buscar en sd['gyro']
    g = sd.get("gyro", {})
    if isinstance(g, (list, tuple)) and len(g) >= 3:
        gdict = {"x": g[0], "y": g[1], "z": g[2]}
    elif isinstance(g, dict):
        gdict = g
    else:
        gdict = {}

    def pick(axis):
        # 1) sd[axis] (caso mixto), 2) gdict[axis], 3) llaves planas 'gyro:x'
        v = sd.get(axis, gdict.get(axis, msg.get(f"gyro:{axis}")))
        return float(v) if v is not None else None

    gx, gy, gz = pick("x"), pick("y"), pick("z")
    if None in (gx, gy, gz):
        return None
    return gx, gy, gz

# --- captura y guardado ---
rows = []          # acumulador de filas
start_time = None  # momento del primer paquete válido

while True:
    data, addr = sock.recvfrom(65535)
    txt = data.decode("utf-8", errors="ignore")
    try:
        msg = json.loads(txt)
    except json.JSONDecodeError:
        # paquete no JSON; ignora
        continue

    g = get_gyro(msg)
    if not g:
        # Imprime solo la sección relevante para depurar (mantengo tu mensaje)
        print("No encuentro gyro en sensordata:", msg.get("sensordata"))
        continue

    gx, gy, gz = g

    # inicializa cronómetro al primer paquete válido
    now = time.time()
    if start_time is None:
        start_time = now
        print(f"⏱️ Iniciando captura por {DURATION_SECONDS}s...")

    # tiempo relativo desde el inicio
    t_rel = now - start_time
    # imprime en consola como antes
    print(f"gyro: x={gx:.6f} y={gy:.6f} z={gz:.6f}")

    # agrega fila (formato largo y claro)
    rows.append({
        "timestamp_iso": datetime.utcnow().isoformat(),
        "t_rel_s": t_rel,
        "sensor": "gyro",
        "x": gx, "y": gy, "z": gz
    })

    # si ya pasó la ventana, guarda y termina
    if t_rel >= DURATION_SECONDS:
        # crear DataFrame y guardar CSV
        df = pd.DataFrame(rows, columns=["timestamp_iso","t_rel_s","sensor","x","y","z"])
        # si existe, sobreescribe; si prefieres con marca de tiempo, cambia OUT_PATH dinámicamente
        os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
        df.to_csv(OUT_PATH, index=False, encoding="utf-8")
        print(f"✅ Guardado {len(df)} filas en: {OUT_PATH}")
        break
