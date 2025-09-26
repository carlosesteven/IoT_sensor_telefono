import socket, json, time, threading
from datetime import datetime
from collections import deque
from flask import Flask, Response, jsonify, render_template_string

# ========= CONFIG =========
UDP_IP = "10.147.20.241"   # IP de tu PC (donde corre Flask)
UDP_PORT = 8080
SENSORS_TO_COLLECT = None  # {"gyro","accel","gps"} para filtrar, o None para todos
MAX_ROWS_MEMORY = 5000     # filas que mantenemos en memoria para mostrar
STREAM_INTERVAL_SEC = 0.2  # frecuencia de actualización del stream SSE
HTTP_HOST = "0.0.0.0"      # cámbialo a "127.0.0.1" si solo verás local
HTTP_PORT = 5050
# =========================

app = Flask(__name__)
rows = deque(maxlen=MAX_ROWS_MEMORY)   # cada fila: dict(timestamp_iso,t_rel_s,sensor,valor)
start_time = None
_start_lock = threading.Lock()

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
    global start_time
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
        t_iso = datetime.utcnow().isoformat()

        for key, val in pairs:
            sensor_name = key.split(":", 1)[0]
            if SENSORS_TO_COLLECT and sensor_name not in SENSORS_TO_COLLECT:
                continue
            try:
                valor = float(val)
            except (TypeError, ValueError):
                continue

            rows.append({
                "timestamp_iso": t_iso,
                "t_rel_s": t_rel,
                "sensor": key,  # p.ej. 'gyro:x', 'gps:latitude'
                "valor": valor
            })

# ---------- SSE stream ----------
@app.route("/stream")
def stream():
    def gen():
        last_len = 0
        while True:
            # si hay nuevas filas, envíalas (en bloque) como una lista JSON
            if len(rows) != last_len:
                payload = list(rows)[last_len:]  # solo lo nuevo
                last_len = len(rows)
                yield f"data: {json.dumps(payload)}\n\n"
            time.sleep(STREAM_INTERVAL_SEC)
    return Response(gen(), mimetype="text/event-stream")

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
// que el contador arranque en lo que ya se pintó (máx 20)
let total = tbody.rows.length;
const es = new EventSource('/stream');

function fmt(n){ return Number(n).toFixed(6); }

es.onmessage = (ev) => {
  try {
    const arr = JSON.parse(ev.data);   // bloque de filas nuevas
    const frag = document.createDocumentFragment();
    for (const row of arr) {
      total++;
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${total}</td>
        <td>${row.timestamp_iso}</td>
        <td>${row.sensor}</td>
        <td>${fmt(row.valor)}</td>
      `;
      frag.appendChild(tr);
    }
    tbody.appendChild(frag);

    // Mantener SOLO las últimas 20 filas visibles
    while (tbody.rows.length > 20) {
      tbody.deleteRow(0); // elimina la más vieja
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
    ultimos = list(rows)[-20:]  # <<< SOLO LOS ÚLTIMOS 20
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