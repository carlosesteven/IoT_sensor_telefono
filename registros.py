#!/usr/bin/env python3
import socket, json, time, os, csv
from collections import deque
from datetime import datetime
import threading

# ============== CONFIG =================
UDP_IP   = "0.0.0.0"     # escucha en todas las interfaces
UDP_PORT = 8080
SAMPLE_HZ = 60           # igual a la tasa de ZIG SIM (p.ej. 60)
CAPTURE_SECONDS = 1.0    # EXACTO 1 segundo de captura
OUT_DIR = "recordings"   # carpeta de salida
# =======================================

LAST = {"acc_x": None, "acc_y": None, "acc_z": None,
        "gyr_x": None, "gyr_y": None, "gyr_z": None}
STOP = False
_ready_evt = threading.Event()
_lock = threading.Lock()

def iter_pairs_from_msg(msg: dict):
    # 1) formato plano "sensor:axis"
    for k, v in list(msg.items()):
        if isinstance(k, str) and ":" in k:
            s, a = k.split(":", 1)
            s, a = s.strip().lower(), a.strip().lower()
            if s in ("accel","accelerometer") and a in ("x","y","z"):
                yield f"acc_{a}", v
            if s in ("gyro","gyroscope") and a in ("x","y","z"):
                yield f"gyr_{a}", v
    # 2) dentro de sensordata
    sd = msg.get("sensordata")
    if isinstance(sd, dict):
        for sensor, payload in sd.items():
            s = str(sensor).lower()
            if s.startswith("acc"):
                pref = "acc"
            elif s.startswith("gyr"):
                pref = "gyr"
            else:
                continue
            if isinstance(payload, dict):
                for axis, val in payload.items():
                    a = str(axis).lower()
                    if a in ("x","y","z"):
                        yield f"{pref}_{a}", val
            elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
                yield f"{pref}_x", payload[0]
                yield f"{pref}_y", payload[1]
                yield f"{pref}_z", payload[2]

def udp_listener():
    """Recibe UDP y actualiza LAST; dispara _ready_evt cuando tengamos los 6 ejes."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(1.0)
    print(f"[UDP] Escuchando en {UDP_IP}:{UDP_PORT}")
    global STOP
    while not STOP:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            continue
        except Exception as e:
            print("[UDP] Error:", e); continue
        try:
            msg = json.loads(data.decode("utf-8", errors="ignore"))
        except Exception:
            continue
        updated = False
        with _lock:
            for k, v in iter_pairs_from_msg(msg):
                try:
                    val = float(v)
                except (TypeError, ValueError):
                    continue
                if k in LAST:
                    LAST[k] = val
                    updated = True
            if updated and all(LAST[k] is not None for k in LAST):
                _ready_evt.set()  # ya tenemos los 6 ejes
    sock.close()

def capture_one_second():
    """Captura EXACTO 1 segundo a SAMPLE_HZ, empezando cuando hay datos listos."""
    # Espera a que lleguen los primeros datos completos
    print("[CAP] Esperando primeros datos (6 ejes)…")
    _ready_evt.wait()  # bloquea hasta tener los 6 ejes al menos una vez
    print("[CAP] Datos detectados. Iniciando captura de 1 segundo…")

    # Prepara archivo
    os.makedirs(OUT_DIR, exist_ok=True)
    ts_name = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    csv_path = os.path.join(OUT_DIR, f"capture_{ts_name}.csv")

    period = 1.0 / SAMPLE_HZ
    n_samples = int(round(CAPTURE_SECONDS * SAMPLE_HZ))
    rows = []

    next_t = time.time()
    for i in range(n_samples):
        # Espera exacta al siguiente tick
        now = time.time()
        if now < next_t:
            time.sleep(max(0, next_t - now))
        next_t += period

        with _lock:
            row = [datetime.utcnow().isoformat(timespec="milliseconds"),
                   LAST["acc_x"], LAST["acc_y"], LAST["acc_z"],
                   LAST["gyr_x"], LAST["gyr_y"], LAST["gyr_z"]]
        rows.append(row)

    # Escribe CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z"])
        w.writerows(rows)

    # Mini resumen
    ax = [r[1] for r in rows]; ay = [r[2] for r in rows]; az = [r[3] for r in rows]
    gx = [r[4] for r in rows]; gy = [r[5] for r in rows]; gz = [r[6] for r in rows]
    def _std(a):
        if not a: return 0.0
        m = sum(a)/len(a)
        return (sum((x-m)**2 for x in a)/len(a))**0.5
    print(f"[OK] Guardado {csv_path}")
    print(f"[SUM] muestras={len(rows)}  Hz≈{SAMPLE_HZ}  "
          f"std_acc≈{(_std(ax)+_std(ay)+_std(az))/3:.4f}  "
          f"std_gyr≈{(_std(gx)+_std(gy)+_std(gz))/3:.4f}")

def main():
    t = threading.Thread(target=udp_listener, daemon=True)
    t.start()
    try:
        capture_one_second()
    finally:
        global STOP
        STOP = True
        time.sleep(0.1)

if __name__ == "__main__":
    main()
