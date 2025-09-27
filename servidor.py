import socket, json, time, threading
from datetime import datetime
from collections import deque
from flask import Flask, Response, jsonify, render_template_string

### >>> NEW: deps para el modelo
import joblib, numpy as np, json as _json

# ========= CONFIG =========
UDP_IP = "10.147.20.241"   # IP de tu PC (donde corre Flask)
UDP_PORT = 8080
SENSORS_TO_COLLECT = None  # {"gyro","accel","gps"} para filtrar, o None para todos
MAX_ROWS_MEMORY = 5000     # filas que mantenemos en memoria para mostrar
STREAM_INTERVAL_SEC = 0.1  # frecuencia de actualización del stream SSE
HTTP_HOST = "0.0.0.0"
HTTP_PORT = 5050

### >>> NEW: modelo entrenado (86%)
MODEL_PATH = "entrenamiento/har_mlp.joblib"
LABELS_JSON = "entrenamiento/har_labels.json"
WINDOW = 128  # tamaño de ventana del UCI HAR

app = Flask(__name__)
rows = deque(maxlen=MAX_ROWS_MEMORY)   # cada fila: dict(timestamp_iso,t_rel_s,sensor,valor)
NEXT_SEQ = 0  # <<< NEW: contador monotónico de filas
start_time = None
_start_lock = threading.Lock()

### >>> NEW: cargar modelo y labels + buffers por eje
MODEL = joblib.load(MODEL_PATH)
with open(LABELS_JSON) as f:
    LABMAP = {int(k): v for k, v in _json.load(f).items()}

BUFF = {
    "acc_x": deque(maxlen=WINDOW), "acc_y": deque(maxlen=WINDOW), "acc_z": deque(maxlen=WINDOW),
    "gyro_x": deque(maxlen=WINDOW), "gyro_y": deque(maxlen=WINDOW), "gyro_z": deque(maxlen=WINDOW),
}

# --- NUEVO: último valor por eje + temporizador de muestreo "cohesionado por tiempo"
LAST = {
    "acc_x": None, "acc_y": None, "acc_z": None,
    "gyro_x": None, "gyro_y": None, "gyro_z": None,
}
SAMPLE_HZ = 60                 # frecuencia a la que quieres tomar muestras para la ventana
SAMPLE_PERIOD = 1.0 / SAMPLE_HZ
_last_sample_t = 0.0
_last_pred_len = 0             # para no repetir predicción en el mismo índice de ventana

# --- NEW: frecuencia de predicción sobre la ventana rodante
PRED_EVERY = 10          # predecir cada 10 muestras nuevas (ajústalo a gusto)
SAMPLE_IDX = 0           # índice global de muestras tomadas
LAST_PRED_IDX = -1       # última muestra en la que se predijo

def _feat_block(arr: np.ndarray):
    """Devuelve [mean, std, min, max, rms, mad] para un array 1D."""
    arr = np.asarray(arr, dtype=float)
    mean = arr.mean()
    std  = arr.std()
    vmin = arr.min()
    vmax = arr.max()
    rms  = np.sqrt((arr**2).mean())
    mad  = np.mean(np.abs(arr - mean))
    return np.array([mean, std, vmin, vmax, rms, mad], dtype=float)

def _make_feature_vector_from_buffers():
    """Concatena features de 6 ejes + magnitudes => (1, 60) o None si no hay 128 muestras aún."""
    if min(len(BUFF[k]) for k in BUFF.keys()) < WINDOW:
        return None
    ax = np.array(BUFF["acc_x"]); ay = np.array(BUFF["acc_y"]); az = np.array(BUFF["acc_z"])
    gx = np.array(BUFF["gyro_x"]); gy = np.array(BUFF["gyro_y"]); gz = np.array(BUFF["gyro_z"])
    acc_mag  = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    feats = np.concatenate([
        _feat_block(ax), _feat_block(ay), _feat_block(az),
        _feat_block(gx), _feat_block(gy), _feat_block(gz),
        _feat_block(acc_mag), _feat_block(gyro_mag),
    ])
    return feats.reshape(1, -1)

def _try_predict_and_push_row(t_iso, t_rel):
    """Si hay ventana completa, predice y agrega una fila 'pred:<LABEL>' con la confianza."""
    X = _make_feature_vector_from_buffers()
    if X is None:
        return
    proba = MODEL.predict_proba(X)[0]
    label_id = int(MODEL.predict(X)[0])
    label = LABMAP.get(label_id, str(label_id))
    conf = float(np.max(proba))
    # Empuja una fila "virtual" con el resultado
    global NEXT_SEQ
    row = {
        "timestamp_iso": t_iso,
        "t_rel_s": t_rel,
        "sensor": f"pred:{label}",
        "valor": conf,
        "seq": NEXT_SEQ  # <<< NEW
    }
    NEXT_SEQ += 1
    rows.append(row)

# ---------- Parser ----------
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

        # 3) ejes sueltos x,y,z (caso mixto)
        if any(k in sd for k in ("x", "y", "z")):
            if "gyro" in sd: base = "gyro"
            elif "accel" in sd: base = "accel"
            else: base = "unknown"
            for axis in ("x", "y", "z"):
                if axis in sd:
                    yield f"{base}:{axis}", sd[axis]

# ---------- UDP listener en hilo ----------
def udp_loop():    
    global start_time, NEXT_SEQ, _last_sample_t, _last_pred_len, SAMPLE_IDX, LAST_PRED_IDX
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"UDP escuchando en {UDP_IP}:{UDP_PORT}")

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
            with _start_lock:
                if start_time is None:
                    start_time = now
                    print("⏱️ Captura iniciada")

        t_rel = now - start_time
        t_iso = datetime.utcnow().isoformat(timespec="milliseconds")

        # --- ACTUALIZA "LAST" con el último valor visto por eje ---
        for key, val in pairs:
            sensor_name = key.split(":", 1)[0]
            if SENSORS_TO_COLLECT and sensor_name not in SENSORS_TO_COLLECT:
                continue
            try:
                valor = float(val)
            except (TypeError, ValueError):
                continue

            # Seguimos mostrando cada lectura como fila normal
            global NEXT_SEQ
            row = {
                "timestamp_iso": t_iso,
                "t_rel_s": t_rel,
                "sensor": key,
                "valor": valor,
                "seq": NEXT_SEQ  # <<< NEW
            }
            NEXT_SEQ += 1
            rows.append(row)

            # Actualiza último valor visto por eje
            if key == "accel:x": LAST["acc_x"] = valor
            elif key == "accel:y": LAST["acc_y"] = valor
            elif key == "accel:z": LAST["acc_z"] = valor
            elif key == "gyro:x":  LAST["gyro_x"] = valor
            elif key == "gyro:y":  LAST["gyro_y"] = valor
            elif key == "gyro:z":  LAST["gyro_z"] = valor

        # --- Si ya tenemos los 6 ejes y pasó el periodo, "tomamos" UNA muestra a la ventana ---
        ready = all(v is not None for v in LAST.values())
        now2 = time.time()
        global _last_sample_t, _last_pred_len

        if ready and (now2 - _last_sample_t) >= SAMPLE_PERIOD:
            # Empuja una muestra (los últimos valores) a los buffers
            BUFF["acc_x"].append(LAST["acc_x"])
            BUFF["acc_y"].append(LAST["acc_y"])
            BUFF["acc_z"].append(LAST["acc_z"])
            BUFF["gyro_x"].append(LAST["gyro_x"])
            BUFF["gyro_y"].append(LAST["gyro_y"])
            BUFF["gyro_z"].append(LAST["gyro_z"])
            _last_sample_t = now2

            SAMPLE_IDX += 1  # <<< NEW

            # Si ya hay 128 por eje y avanzó el índice, predice y agrega fila 'pred:<LABEL>'
            min_len = min(len(BUFF[k]) for k in BUFF.keys())
            if min_len >= WINDOW and (SAMPLE_IDX - LAST_PRED_IDX) >= PRED_EVERY:
                _try_predict_and_push_row(t_iso, t_rel)
                LAST_PRED_IDX = SAMPLE_IDX  # <<< NEW

# ---------- SSE stream ----------
@app.route("/stream")
def stream():
    def gen():
        last_seq = -1  # <<< NEW: último seq enviado a este cliente
        while True:
            snapshot = list(rows)  # toma foto de las filas actuales
            # filtra solo lo nuevo desde el último seq enviado
            new_rows = [r for r in snapshot if r.get("seq", -1) > last_seq]
            if new_rows:
                last_seq = new_rows[-1]["seq"]
                yield f"data: {json.dumps(new_rows)}\n\n"
            time.sleep(STREAM_INTERVAL_SEC)
    return Response(gen(), mimetype="text/event-stream")

@app.route("/reset", methods=["GET"])
def reset():
    global NEXT_SEQ, start_time, _last_sample_t, _last_pred_len
    rows.clear()
    for dq in BUFF.values(): dq.clear()
    for k in LAST: LAST[k] = None
    start_time = None
    _last_sample_t = 0.0
    _last_pred_len = 0
    NEXT_SEQ = 0
    return jsonify(ok=True)

# ---------- API simple (útil para debug) ----------
@app.route("/api/latest")
def api_latest():
    return jsonify(list(rows)[-100:])  # últimas 100 filas

# ---------- UI ----------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8" />
<title>Sensor Monitor (UDP → Flask)</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  body { font-family: system-ui, sans-serif; margin: 20px; background: #0b0b0b; color: #e5ffe5;}
  h1 { margin: 0 0 8px; font-size: 22px; }
  .muted { color: #9acd9a; font-size: 12px; margin-bottom: 12px;}
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 8px 10px; border-bottom: 1px solid #1f3a1f; }
  th { text-align: left; background: #123; color:#cfe;}
  tr:nth-child(even) td { background: #0f1a0f; }
  #count { font-weight: bold; }
</style>
</head>
<body>
  <h1>Sensor Monitor</h1>
  <div class="muted">Escuchando UDP en <code>{{udp_ip}}</code>:<code>{{udp_port}}</code>. Stream SSE activo.</div>
  <div class="muted">Filtros: <code>{{filter}}</code></div>
  <div class="muted">Actividad: <b id="act">—</b> <span id="conf"></span></div>

  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Timestamp</th>
        <th>Sensor</th>
        <th>Valor</th>
      </tr>
    </thead>
    <tbody id="tbody">
{% for r in ultimos %}
  <tr>
    <td>{{ loop.index }}</td>
    <td>{{ r.timestamp_iso }}</td>
    <td>{{ r.sensor }}</td>
    <td>{{ "%.6f"|format(r.valor) }}</td>
  </tr>
{% endfor %}
</tbody>
</table>

<script>
const tbody = document.getElementById('tbody');
let total = tbody.rows.length;
const es = new EventSource('/stream');

const act = document.getElementById('act');
const conf = document.getElementById('conf');

function fmt(n){ return Number(n).toFixed(6); }

es.onmessage = (ev) => {
  try {
    const arr = JSON.parse(ev.data);
    const frag = document.createDocumentFragment();
    for (const row of arr) {
      // Si es una predicción, actualiza encabezado
      if (typeof row.sensor === 'string' && row.sensor.startsWith('pred:')) {
        const label = row.sensor.slice(5); // quita 'pred:'
        act.textContent = label;
        if (typeof row.valor === 'number') {
          conf.textContent = `(${(row.valor*100).toFixed(1)}%)`;
        } else {
          conf.textContent = '';
        }
      }

      // Agregar fila a la tabla
      total++;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${total}</td>
        <td>${row.timestamp_iso}</td>
        <td>${row.sensor}</td>
        <td>${typeof row.valor === 'number' ? fmt(row.valor) : row.valor}</td>
      `;
      frag.appendChild(tr);
    }
    tbody.appendChild(frag);

    // Mantener SOLO las últimas 20 filas visibles
    while (tbody.rows.length > 20) {
      tbody.deleteRow(0);
    }
  } catch (e) {
    console.error('parse error', e);
  }
};
</script>
</body>
</html>
"""

@app.route("/")
def index():
    filt = ",".join(sorted(SENSORS_TO_COLLECT)) if SENSORS_TO_COLLECT else "NINGUNO"
    ultimos = list(rows)[-20:]
    return render_template_string(
        INDEX_HTML,
        udp_ip=UDP_IP,
        udp_port=UDP_PORT,
        filter=filt,
        ultimos=ultimos
    )

# ---------- main ----------
if __name__ == "__main__":
    t = threading.Thread(target=udp_loop, daemon=True)
    t.start()
    print(f"HTTP en http://{HTTP_HOST}:{HTTP_PORT}")
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False, threaded=True)