#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import socket
import struct
import json
import time
import threading
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, render_template


# ==================== CONFIG ====================
UDP_HOST = "0.0.0.0"      # escucha en todas las interfaces
UDP_PORT = 8080           # puerto donde Serial Sensor envía
WINDOW_SEC = 1.0          # duración de ventana en segundos
STEP_SEC = 0.5            # cada cuánto predecir
MODEL_PATH = "IA_Movimientos.joblib"  # ruta fija a tu modelo

HTTP_HOST = "0.0.0.0"
HTTP_PORT = 5050
POLL_MS   = 500           # refresco del frontend

# ==================== FEATURES ====================
def ingenieriaCaracteristicas(df: pd.DataFrame):
    F1 = df['Ax'].mean()
    F2 = df['Ay'].mean()
    F3 = df['Az'].mean()
    F4 = (df['Ax'].std() + df['Ay'].std() + df['Az'].std()) / 3.0
    F5 = np.sqrt(df['Ax']**2 + df['Ay']**2 + df['Az']**2).mean()
    return [float(F1), float(F2), float(F3), float(F4), float(F5)]

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
                if isinstance(payload, dict):
                    for axis, val in payload.items():
                        a = str(axis).lower()
                        if a in ("x","y","z"):
                            yield "acc", a, val
                elif isinstance(payload, (list, tuple)) and len(payload) >= 3:
                    for a, val in zip(("x","y","z"), payload[:3]):
                        yield "acc", a, val

# ==================== ESTADO COMPARTIDO ====================
state_lock = threading.Lock()
window = deque()  # (t, Ax, Ay, Az)
last_pred = {
    "label": None,            # "R" o "L"
    "prob_rapido": None,      # float 0..1
    "F": [None]*5,            # F1..F5
    "n": 0,                   # muestras en ventana
    "updated_at": None,       # timestamp texto
}

# ==================== MODELO ====================
model = None
f5_threshold_default = 13.1425
try:
    model = joblib.load(MODEL_PATH)
    print(f"[MODEL] Modelo cargado: {MODEL_PATH}")
except Exception as e:
    print(f"[MODEL] No se pudo cargar {MODEL_PATH}: {e}. Usaré umbral F5 por defecto {f5_threshold_default}.")

# ==================== HILO UDP + PREDICCIÓN ====================
def udp_loop():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(1.0)
    print(f"[UDP] Escuchando en {UDP_HOST}:{UDP_PORT}")

    last_pred_t = 0.0
    try:
        while True:
            now = time.time()
            # Purga ventana
            cutoff = now - WINDOW_SEC
            with state_lock:
                while window and window[0][0] < cutoff:
                    window.popleft()

            # Recibir
            try:
                data, _ = sock.recvfrom(65535)
            except socket.timeout:
                data = None

            if data:
                if len(data) == MESSAGE_LENGTH:
                    decoded = decode_serialsensor_binary(data)
                    if decoded:
                        _, x, y, z = decoded
                        with state_lock:
                            window.append((now, x, y, z))
                elif data[:1] in (b'{', b'['):
                    try:
                        msg = json.loads(data.decode("utf-8", errors="ignore"))
                    except Exception:
                        msg = None
                    if isinstance(msg, dict):
                        axes = {"x": None, "y": None, "z": None}
                        for pref, axis, val in iter_pairs_from_json_msg(msg):
                            if pref == "acc":
                                try:
                                    axes[axis] = float(val)
                                except (TypeError, ValueError):
                                    pass
                        if None not in axes.values():
                            with state_lock:
                                window.append((now, axes["x"], axes["y"], axes["z"]))

            # Predicción periódica
            if (now - last_pred_t) >= STEP_SEC:
                last_pred_t = now
                with state_lock:
                    n = len(window)
                    if n >= 3:
                        arr = np.array([[t, ax, ay, az] for (t, ax, ay, az) in window], dtype=float)
                        df_win = pd.DataFrame(arr[:, 1:], columns=["Ax","Ay","Az"])
                        F = ingenieriaCaracteristicas(df_win)

                        if model is not None:
                            try:
                                prob = float(model.predict_proba([F])[0, 1])
                                label = "R" if prob >= 0.5 else "L"
                            except Exception as e:
                                # fallback
                                prob = 1.0 / (1.0 + np.exp(-(F[4] - f5_threshold_default)))
                                label = "R" if F[4] > f5_threshold_default else "L"
                        else:
                            prob = 1.0 / (1.0 + np.exp(-(F[4] - f5_threshold_default)))
                            label = "R" if F[4] > f5_threshold_default else "L"

                        last_pred.update({
                            "label": label,
                            "prob_rapido": prob,
                            "F": [round(x, 6) for x in F],
                            "n": n,
                            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
    finally:
        try:
            sock.close()
        except Exception:
            pass

# ==================== APP WEB (Flask) ====================
app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="es">
  <head>
    <meta charset="utf-8">
    <title>Movimiento Rápido/Lento • Realtime</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <!-- Bootstrap 4 -->
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.6.2/css/bootstrap.min.css">
    <!-- Font Awesome 4 -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>
      body { background:#0f1115; color:#eaeef2; }
      .card { background:#151922; border:none; }
      .badge-lg { font-size:1.2rem; }
      .title { letter-spacing:.3px; }
      .value { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
      .icon-big { font-size:3rem; vertical-align:middle; }
      .label-R { color:#ff6b6b; }
      .label-L { color:#2ecc71; }
      .footer { color:#aab2bd; font-size: .9rem; }
      .table td, .table th { border-color:#232a36 !important; }
    </style>
  </head>
  <body>
    <div class="container py-4">
      <div class="d-flex align-items-center mb-4">
        <i class="fa fa-bolt icon-big mr-3"></i>
        <h3 class="m-0 title">Clasificador de Movimiento – Realtime</h3>
      </div>

      <div class="row">
        <div class="col-lg-5 mb-3">
          <div class="card p-3">
            <div class="d-flex align-items-center justify-content-between">
              <div>
                <div class="text-muted">Estado actual</div>
                <div id="pred_label" class="h2 mb-1 label-L">Lento</div>
                <div class="text-muted">
                  <i class="fa fa-clock-o"></i> <span id="updated_at">—</span>
                </div>
              </div>
              <div class="text-right">
                <div class="mb-2">
                  <span class="badge badge-pill badge-info badge-lg">
                    <i class="fa fa-database"></i> n=<span id="n">0</span>
                  </span>
                </div>
                <div>
                  <span class="badge badge-pill badge-secondary badge-lg">
                    <i class="fa fa-random"></i> Prob. rápido:
                    <span class="value" id="prob">0.000</span>
                  </span>
                </div>
              </div>
            </div>
            <hr>
            <div class="small footer">
              <i class="fa fa-info-circle"></i>
              Ventana {{win}}s • Paso {{step}}s • UDP {{udp_host}}:{{udp_port}} • Modelo: {{model_path}}
            </div>
          </div>
        </div>

        <div class="col-lg-7 mb-3">
          <div class="card p-3">
            <div class="d-flex align-items-center mb-2">
              <i class="fa fa-sliders icon-big mr-3"></i>
              <h5 class="m-0">Características (F1–F5)</h5>
            </div>
            <table class="table table-sm table-dark">
              <thead>
                <tr><th>Feature</th><th>Valor</th></tr>
              </thead>
              <tbody>
                <tr><td>F1 = mean(Ax)</td><td class="value" id="f1">—</td></tr>
                <tr><td>F2 = mean(Ay)</td><td class="value" id="f2">—</td></tr>
                <tr><td>F3 = mean(Az)</td><td class="value" id="f3">—</td></tr>
                <tr><td>F4 = mean(std(Ax,Ay,Az))</td><td class="value" id="f4">—</td></tr>
                <tr><td>F5 = mean(|A|)</td><td class="value" id="f5">—</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div class="text-center text-muted mt-3 footer">
        <i class="fa fa-heart"></i> Realtime dashboard • Refresh {{poll}} ms
      </div>
    </div>

    <script>
      function refresh() {
        fetch('/api/status')
          .then(r => r.json())
          .then(s => {
            const elLabel = document.getElementById('pred_label');
            const elProb  = document.getElementById('prob');
            const elN     = document.getElementById('n');
            const elTime  = document.getElementById('updated_at');
            const f1 = document.getElementById('f1');
            const f2 = document.getElementById('f2');
            const f3 = document.getElementById('f3');
            const f4 = document.getElementById('f4');
            const f5 = document.getElementById('f5');

            elLabel.textContent = s.label || '—';
            elLabel.classList.remove('label-R','label-L');
            elLabel.classList.add(s.label === 'R' ? 'label-R' : 'label-L');
            elProb.textContent = (s.prob_rapido !== null) ? s.prob_rapido.toFixed(3) : '—';
            elN.textContent = s.n || 0;
            elTime.textContent = s.updated_at || '—';
            const F = s.F || [null,null,null,null,null];
            f1.textContent = F[0] !== null ? F[0].toFixed(3) : '—';
            f2.textContent = F[1] !== null ? F[1].toFixed(3) : '—';
            f3.textContent = F[2] !== null ? F[2].toFixed(3) : '—';
            f4.textContent = F[3] !== null ? F[3].toFixed(3) : '—';
            f5.textContent = F[4] !== null ? F[4].toFixed(3) : '—';
          })
          .catch(() => {});
      }
      setInterval(refresh, {{poll}});
      refresh();
    </script>
  </body>
</html>
"""

@app.route("/")
def index():
    return render_template(
        "dashboard.html",
        win=WINDOW_SEC,
        step=STEP_SEC,
        udp_host=UDP_HOST,
        udp_port=UDP_PORT,
        model_path=MODEL_PATH,
        poll=POLL_MS
    )

@app.route("/api/status")
def api_status():
    with state_lock:
        payload = dict(last_pred)
    return jsonify(payload)

# ==================== ENTRYPOINT ====================
if __name__ == "__main__":
    # Lanzar el hilo del listener UDP
    t = threading.Thread(target=udp_loop, daemon=True)
    t.start()
    # Levantar servidor web
    print(f"[WEB] http://{HTTP_HOST}:{HTTP_PORT}")
    app.run(host=HTTP_HOST, port=HTTP_PORT, debug=False)
