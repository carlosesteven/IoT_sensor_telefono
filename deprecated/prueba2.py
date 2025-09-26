import socket, json

UDP_IP = "10.147.20.189"
UDP_PORT = 8888

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

first = True
while True:
    data, addr = sock.recvfrom(65535)
    txt = data.decode("utf-8", errors="ignore")
    try:
        msg = json.loads(txt)
    except json.JSONDecodeError:
        print("Paquete no-JSON:", repr(txt[:120]))
        continue

    g = get_gyro(msg)
    if not g:
        # Imprime solo la sección relevante para depurar
        print("No encuentro gyro en sensordata:", msg.get("sensordata"))
        continue

    gx, gy, gz = g
    print(f"gyro: x={gx:.6f} y={gy:.6f} z={gz:.6f}")
