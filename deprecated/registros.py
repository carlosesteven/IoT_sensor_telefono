#!/usr/bin/env python3
import socket, json, time, os, csv, sys, threading, re
from datetime import datetime

# ============== CONFIG =================
UDP_IP   = "0.0.0.0"     # escucha en todas las interfaces
UDP_PORT = 8080
SAMPLE_HZ = 60           # igual a la tasa de ZIG SIM (p.ej. 60)
CAPTURE_SECONDS = 1.0    # EXACTO 1 segundo por archivo
OUT_DIR = "recordings"   # carpeta de salida
# =======================================

# Últimos valores recibidos por eje
LAST = {"acc_x": None, "acc_y": None, "acc_z": None,
        "gyr_x": None, "gyr_y": None, "gyr_z": None}

STOP = False
_ready_evt = threading.Event()
_lock = threading.Lock()

# Tag para rotular la captura (L=lento, R=rápido). Cambiable desde teclado.
CURRENT_TAG = "L"

def iter_pairs_from_msg(msg: dict):
    """Normaliza diferentes formatos (plano o 'sensordata') en claves acc_* / gyr_*."""
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

def next_index_from_disk():
    """Lee OUT_DIR y encuentra el próximo índice vN a usar."""
    os.makedirs(OUT_DIR, exist_ok=True)
    pat = re.compile(r"^v(\d+)(?:_[LR])?\.csv$")
    mx = 0
    for name in os.listdir(OUT_DIR):
        m = pat.match(name)
        if m:
            try:
                mx = max(mx, int(m.group(1)))
            except ValueError:
                pass
    return mx + 1

def capture_block(idx: int, tag: str):
    """Captura EXACTO CAPTURE_SECONDS a SAMPLE_HZ y escribe recordings/v{idx}_{tag}.csv"""
    period = 1.0 / SAMPLE_HZ
    n_samples = int(round(CAPTURE_SECONDS * SAMPLE_HZ))
    rows = []

    start_t = time.time()
    next_t = start_t
    for _ in range(n_samples):
        # Espera exacta al siguiente tick
        now = time.time()
        if now < next_t:
            time.sleep(max(0, next_t - now))
        next_t += period

        with _lock:
            row = [datetime.utcnow().isoformat(timespec="milliseconds"),
                   LAST["acc_x"], LAST["acc_y"], LAST["acc_z"],
                   LAST["gyr_x"], LAST["gyr_y"], LAST["gyr_z"],
                   tag]
        rows.append(row)

    # Escribe CSV
    fname = f"v{idx}_{tag}.csv"
    csv_path = os.path.join(OUT_DIR, fname)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","acc_x","acc_y","acc_z","gyr_x","gyr_y","gyr_z","LR"])
        w.writerows(rows)

    # Pequeño resumen
    print(f"[OK] {fname}  ({len(rows)} muestras, tag={tag})")
    return csv_path

def input_monitor():
    """Hilo que permite cambiar el tag y salir:
       - 'l' + Enter -> tag = L
       - 'r' + Enter -> tag = R
       - 'q' + Enter -> terminar
    """
    global CURRENT_TAG, STOP
    print("[KEYS] Escribe 'l' (lento), 'r' (rápido) o 'q' (salir) y Enter.")
    while not STOP:
        try:
            line = sys.stdin.readline()
        except Exception:
            break
        if not line:
            continue
        cmd = line.strip().lower()
        if cmd == 'l':
            CURRENT_TAG = "L"
            print("[TAG] Cambiado a L (lento)")
        elif cmd == 'r':
            CURRENT_TAG = "R"
            print("[TAG] Cambiado a R (rápido)")
        elif cmd == 'q':
            print("[EXIT] Saliendo…")
            STOP = True
            break

def main():
    global STOP  # 👈 aquí, al inicio

    # Hilo UDP
    t_udp = threading.Thread(target=udp_listener, daemon=True)
    t_udp.start()

    # Hilo entrada teclado
    t_in = threading.Thread(target=input_monitor, daemon=True)
    t_in.start()

    # Espera primeros datos completos
    print("[CAP] Esperando primeros datos (6 ejes)…")
    _ready_evt.wait()
    print("[CAP] Datos detectados. Comenzando captura continua de 1s/archivo…")

    idx = next_index_from_disk()
    try:
        while not STOP:
            tag = CURRENT_TAG  # congelamos el tag al inicio del bloque
            capture_block(idx, tag)
            idx += 1
            # inmediatamente inicia el siguiente bloque de 1s
    except KeyboardInterrupt:
        pass
    finally:
        STOP = True
        time.sleep(0.1)

if __name__ == "__main__":
    main()