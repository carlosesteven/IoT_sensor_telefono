import socket
import struct
import errno
import threading
from flask import Flask, jsonify
 
# General config
UDP_IP = "192.168.1.34"
UDP_PORT = 8080
MESSAGE_LENGTH = 13
 
# Global variable to hold the latest sensor data
latest_sensor_data = {
    'ax': 0.0,
    'ay': 0.0,
    'az': 0.0
}
 
# Flask app setup
app = Flask(__name__)
 
# UDP listener function
def udp_listener():
    print("This PC's IP: ", UDP_IP)
    print("Listening on Port: ", UDP_PORT)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False)
    sock.bind((UDP_IP, UDP_PORT))
 
    while True:
        keepReceiving = True
        newestData = None
        while keepReceiving:
            try:
                data, fromAddr = sock.recvfrom(MESSAGE_LENGTH)
                if data:
                    newestData = data
            except socket.error as why:
                if why.args[0] == errno.EWOULDBLOCK:
                    keepReceiving = False
                else:
                    raise why
 
        if newestData is not None:
            # Unpack the data and update the global dictionary
            global latest_sensor_data
            ax_tuple = struct.unpack_from('<f', newestData, 1)
            ay_tuple = struct.unpack_from('<f', newestData, 5)
            az_tuple = struct.unpack_from('<f', newestData, 9)
 
            latest_sensor_data['ax'] = ax_tuple[0]
            latest_sensor_data['ay'] = ay_tuple[0]
            latest_sensor_data['az'] = az_tuple[0]
            print("Received data: ", latest_sensor_data)
 
 
# Flask endpoint to serve the data
@app.route('/sensor_data', methods=['GET'])
def get_sensor_data():
    return jsonify(latest_sensor_data)
 
# Main part of the script
if __name__ == '__main__':
    # Start the UDP listener in a separate thread
    listener_thread = threading.Thread(target=udp_listener, daemon=True)
    listener_thread.start()
 
    # Run the Flask app
    app.run(host='0.0.0.0', port=5050)