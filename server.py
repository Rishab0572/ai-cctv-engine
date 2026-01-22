"""
stream_server.py
Serves your Mac webcam as an MJPEG HTTP stream with Basic Auth.

Usage:
  python stream_server.py --user demo --pass secret
Then open: http://<your_ip>:5000/stream
You can test via VLC or the detection client.
"""

from flask import Flask, Response, request, abort
import cv2
import argparse
import socket

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--user", default="demo", help="username for basic auth")
parser.add_argument("--passw", default="secret", help="password for basic auth")
parser.add_argument("--camera", type=int, default=0, help="camera index")
parser.add_argument("--port", type=int, default=5003)
args = parser.parse_args()

USERNAME = args.user
PASSWORD = args.passw
CAM_INDEX = args.camera

# Helper: get local IP
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't need to be reachable
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

# Video capture (webcam)
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise SystemExit("Cannot open webcam (index {}).".format(CAM_INDEX))

def check_auth():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Basic "):
        return False
    import base64
    token = auth.split(" ", 1)[1].strip()
    try:
        decoded = base64.b64decode(token).decode("utf-8")
    except Exception:
        return False
    parts = decoded.split(":", 1)
    if len(parts) != 2:
        return False
    return parts[0] == USERNAME and parts[1] == PASSWORD

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        # encode as jpeg
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        # multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/stream')
def stream():
    # enforce basic auth
    if not check_auth():
        # send 401 with WWW-Authenticate so clients ask for creds
        return Response('Unauthorized', status=401, headers={'WWW-Authenticate': 'Basic realm="Login Required"'})
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    ip = get_local_ip()
    return f"Stream is available at http://{ip}:{args.port}/stream (user: {USERNAME})"

if __name__ == '__main__':
    ip = get_local_ip()
    print(f"Serving webcam at http://{ip}:{args.port}/stream (user: {USERNAME}, pass: {PASSWORD})")
    app.run(host='0.0.0.0', port=args.port, threaded=True)