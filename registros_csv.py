#!/usr/bin/env python3
import socket, json, time, os, csv, sys, threading, re, struct
from datetime import datetime
from collections import deque

# ============== CONFIG =================
UDP_IP   = "0.0.0.0"     # escucha en todas las interfaces
UDP_PORT = 8080
CAPTURE_SECONDS = 1.0    # 1 segundo por archivo (sin límite de muestras)
OUT_DIR = "recordings"   # carpeta de salida

# Sensores requeridos para arrancar a escribir (no limita lo que se guarda)
REQUIRE_SENSORS = {"acc"}  # usa {"acc","grav"} si quieres esperar ambos antes de empezar
# =======================================

MESSAGE_LENGTH = 13  # 1 byte tipo + 3*4 bytes float LE

# Mapa de tags -> sensor. Ajusta si tu app usa otros códigos para el byte 0.
SENSOR_TAG_MAP = {
    ord('A'): "acc",  ord('a'): "acc",     # Accelerometer (total)
    ord('G'): "gyr",  ord('g'): "gyr",     # Gyroscope
    ord('V'): "grav", ord('v'): "grav",    # Gravity (vector)
    1: "acc",
    2: "gyr",
    3: "grav",
}

LAST = {
    "acc_x": None,  "acc_y": None,  "acc_z": None,
    "grav_x": None, "grav_y": None, "grav_z": None,
    "gyr_x": None,  "gyr_y": None,  "gyr_z": None,
}

STOP = False
_ready_evt = threading.Event()
_lock = threading.Lock()

# Buffer de paquetes ya decodificados: (timestamp_iso, sensor, x, y, z)
BUFFER = deque()

CURRENT_TAG = "L"  # 'L' (lento) / 'R' (rápido)

def iter_pairs_from_json_msg(msg: dict):
    for k, v in list(msg.items()):
        if isinstance(k, str) and ":" in k:
            s, a = k.split(":", 1)
            s, a = s.strip().lower(), a.strip().lower()
            if s in ("accel","accelerometer","acc") and a in ("x","y","z"):
                yield "acc", a, v
            if s in ("gyro","gyroscope","gyr") and a in ("x","y","z"):
                yield "gyr", a, v
            if s in ("gravity","grav") and a in ("x","y","z"):
                yield "grav", a, v
    sd = msg.get("sensordata")
    if isinstance(sd, dict):
        for sensor, payload in sd.items():
            s = str(sensor).lower()
            if s.startswith(("accel","acc")):
                pref = "acc"
            elif s.startswith(("gyro","gyr")):
                pref = "gyr"
            elif s.startswith(("gravity","grav")):
                pref = "grav"
            else:
                continue
            if isinstance(payload, dict):
                for axis, val in payload.items():
                    a = str(axis).lower()
                    if a in ("x","y","z"):
                        yield pref, a, val
            elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
                for a, val in zip(("x","y","z"), payload[:3]):
                    yield pref, a, val

def decode_serialsensor_binary(pkt: bytes):
    """Devuelve (sensor, x, y, z) o None."""
    if len(pkt) != MESSAGE_LENGTH:
        return None
    tag = pkt[0]
    pref = SENSOR_TAG_MAP.get(tag)
    if pref is None and 32 <= tag <= 126:
        pref = SENSOR_TAG_MAP.get(ord(chr(tag)))
    if pref not in ("acc", "gyr", "grav"):
        return None
    try:
        x = struct.unpack_from('<f', pkt, 1)[0]
        y = struct.unpack_from('<f', pkt, 5)[0]
        z = struct.unpack_from('<f', pkt, 9)[0]
        return (pref, float(x), float(y), float(z))
    except Exception:
        return None

def have_required_sensors():
    need = []
    if "acc" in REQUIRE_SENSORS:
        need += ["acc_x","acc_y","acc_z"]
    if "grav" in REQUIRE_SENSORS:
        need += ["grav_x","grav_y","grav_z"]
    if "gyr" in REQUIRE_SENSORS:
        need += ["gyr_x","gyr_y","gyr_z"]
    return all(LAST[k] is not None for k in need)

def udp_listener():
    """Recibe UDP, actualiza LAST y apila cada muestra decodificada en BUFFER."""
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

        iso_ts = datetime.utcnow().isoformat(timespec="milliseconds")

        # Binario 13B
        if len(data) == MESSAGE_LENGTH:
            decoded = decode_serialsensor_binary(data)
            if decoded:
                sensor, x, y, z = decoded
                with _lock:
                    if sensor == "acc":
                        LAST["acc_x"], LAST["acc_y"], LAST["acc_z"] = x, y, z
                    elif sensor == "gyr":
                        LAST["gyr_x"], LAST["gyr_y"], LAST["gyr_z"] = x, y, z
                    elif sensor == "grav":
                        LAST["grav_x"], LAST["grav_y"], LAST["grav_z"] = x, y, z
                    BUFFER.append((iso_ts, sensor, x, y, z))
                    if have_required_sensors():
                        _ready_evt.set()
            continue

        # JSON
        if data and data[:1] in (b'{', b'['):
            try:
                msg = json.loads(data.decode("utf-8", errors="ignore"))
            except Exception:
                continue
            updates = {}
            for pref, axis, val in iter_pairs_from_json_msg(msg):
                try:
                    fv = float(val)
                except (TypeError, ValueError):
                    continue
                updates.setdefault(pref, {})[axis] = fv
            if not updates:
                continue
            with _lock:
                for pref, axes in updates.items():
                    x = axes.get("x"); y = axes.get("y"); z = axes.get("z")
                    if x is None or y is None or z is None:
                        continue
                    if pref == "acc":
                        LAST["acc_x"], LAST["acc_y"], LAST["acc_z"] = x, y, z
                    elif pref == "gyr":
                        LAST["gyr_x"], LAST["gyr_y"], LAST["gyr_z"] = x, y, z
                    elif pref == "grav":
                        LAST["grav_x"], LAST["grav_y"], LAST["grav_z"] = x, y, z
                    BUFFER.append((iso_ts, pref, x, y, z))
                if have_required_sensors():
                    _ready_evt.set()
    sock.close()

def next_index_from_disk():
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

def drain_buffer_rows():
    """Saca todo lo acumulado en BUFFER y devuelve filas CSV:
       [timestamp, sensor, x, y, z, LR]"""
    rows = []
    with _lock:
        while BUFFER:
            ts, sensor, x, y, z = BUFFER.popleft()
            #rows.append([ts, sensor, x, y, z, CURRENT_TAG])
            rows.append([ts, x, y, z, CURRENT_TAG])
    return rows

def capture_every_second():
    """Cada segundo: toma TODO lo recibido en ese intervalo y lo vuelca a un CSV.
       No crea archivo si no hubo datos."""
    idx = next_index_from_disk()
    try:
        while not STOP:
            t0 = time.time()
            while True:
                if STOP:
                    return
                if time.time() - t0 >= CAPTURE_SECONDS:
                    break
                time.sleep(0.01)

            rows = drain_buffer_rows()
            if not rows:
                # No se escribe nada si no hubo paquetes en este segundo
                continue

            fname = f"{CURRENT_TAG}{idx}.csv"
            csv_path = os.path.join(OUT_DIR, fname)
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                #w.writerow(["t","Sensor","Ax","Ay","Az","Label"])
                w.writerow(["t","Ax","Ay","Az","Label"])
                w.writerows(rows)
            print(f"[OK] {fname}  ({len(rows)} paquetes)")
            idx += 1
    except KeyboardInterrupt:
        pass

def input_monitor():
    """l = Lento | r = Rápido | q = Salir"""
    global STOP, CURRENT_TAG
    print("[KEYS] Escribe 'l' (lento), 'r' (rápido) o 'q' (salir).")
    while not STOP:
        line = sys.stdin.readline()
        if not line:
            continue
        cmd = line.strip().lower()
        if cmd == 'l':
            CURRENT_TAG = "L"; print("[TAG] L")
        elif cmd == 'r':
            CURRENT_TAG = "R"; print("[TAG] R")
        elif cmd == 'q':
            print("[EXIT] Saliendo…"); STOP = True; break

def main():
    global STOP
    t_udp = threading.Thread(target=udp_listener, daemon=True); t_udp.start()
    t_in  = threading.Thread(target=input_monitor, daemon=True); t_in.start()

    req = "+".join(sorted(REQUIRE_SENSORS))
    print(f"[CAP] Esperando primeros datos requeridos ({req})…")
    _ready_evt.wait()
    print("[CAP] Iniciando archivos de 1s con TODO lo recibido (sin límite de muestras)…")

    capture_every_second()
    STOP = True
    time.sleep(0.1)

if __name__ == "__main__":
    main()