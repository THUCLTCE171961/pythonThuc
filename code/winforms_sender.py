import socket
import struct
import cv2

def open_camera_socket(host='127.0.0.1', port=6000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        print("Connected to C# camera receiver!")
    except Exception as e:
        print("Không kết nối được tới C#: ", e)
        s = None
    return s

def send_frame_to_winforms(s, frame):
    if s is None:
        return
    result, img_encoded = cv2.imencode('.jpg', frame)
    if not result:
        print("Lỗi encode JPEG.")
        return
    data = img_encoded.tobytes()
    try:
        s.sendall(struct.pack(">L", len(data)) + data)
    except Exception as e:
        pass
def send_command_to_winforms(gesture_name, host='127.0.0.1', port=5006):
    """
    Gửi tên gesture qua TCP socket tới server.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        s.sendall(gesture_name.encode())
        resp = s.recv(1024)
        print("Phản hồi từ WinForms:", resp.decode())
        s.close()
    except Exception as e:
        print("Lỗi gửi lệnh tới WinForms:", e)

def receive_gesture_name(host='127.0.0.1', port=7000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    print("Đang chờ tên gesture từ WinForms...")
    conn, addr = s.accept()
    gesture_name = conn.recv(1024).decode().strip()
    conn.sendall(b'OK')
    conn.close()
    s.close()
    print(f"Nhận tên gesture: {gesture_name}")
    return gesture_name

def start_camera_server(host='127.0.0.1', port=6001):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"[PYTHON] Camera server đang lắng nghe ở {host}:{port} ...")
    conn, addr = server.accept()
    print(f"[PYTHON] WinForms đã kết nối: {addr}")
    return conn

# def send_frame_to_winforms(conn, frame):
#     result, img_encoded = cv2.imencode('.jpg', frame)
#     if not result:
#         print("[PYTHON] Lỗi encode ảnh!")
#         return
#     data = img_encoded.tobytes()
#     print(f"[PYTHON] Đang gửi frame, kích thước: {len(data)} bytes")
#     try:
#         conn.sendall(struct.pack(">L", len(data)) + data)
#     except Exception as e:
#         print("[PYTHON] Lỗi khi gửi frame:", e)

def start_status_server(host='127.0.0.1', port=6002):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)
    print(f"[PYTHON] Status server đang lắng nghe ở {host}:{port} ...")
    conn, addr = server.accept()
    print(f"[PYTHON] WinForms đã kết nối (status): {addr}")
    return conn

def send_status_to_winforms(conn, state, pose, saved):
    text = f"{state}|{pose}|{saved}"
    try:
        conn.sendall(text.encode())
    except Exception as e:
        print("[PYTHON] Lỗi khi gửi status:", e)

def receive_pose_name(host='127.0.0.1', port=7000):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(1)
    print("Waiting for pose name from WinForms...")
    conn, addr = s.accept()
    pose_name = conn.recv(1024).decode().strip()
    conn.sendall(b'OK')
    conn.close()
    s.close()
    print(f"Received pose name: {pose_name}")
    return pose_name