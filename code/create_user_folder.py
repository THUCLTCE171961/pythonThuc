# import os
# import shutil
# import sys

# def create_user_folder_from_defaults(user_id, base_dir):
#     defaults_dir = os.path.join(base_dir, 'defaults')
#     user_folder = os.path.join(base_dir, f'user_{user_id}')
#     if not os.path.exists(user_folder):
#         os.makedirs(user_folder)
#         print(f"[INFO] Created user folder: {user_folder}")
#     else:
#         print(f"[INFO] User folder already exists: {user_folder}")
#     if not os.path.exists(defaults_dir):
#         print(f"[ERROR] Defaults folder not found: {defaults_dir}")
#         return
#     for item in os.listdir(defaults_dir):
#         s = os.path.join(defaults_dir, item)
#         d = os.path.join(user_folder, item)
#         if os.path.isdir(s):
#             shutil.copytree(s, d, dirs_exist_ok=True)  # Python >= 3.8
#             print(f"[INFO] Copied folder: {item}")
#         else:
#             shutil.copy2(s, d)
#             print(f"[INFO] Copied file: {item}")
#     print(f"[SUCCESS] All defaults copied to {user_folder}")

# if __name__ == '__main__':
#     if len(sys.argv) < 2:
#         print("Usage: python create_user_folder.py [user_id]")
#         sys.exit(1)
#     user_id = sys.argv[1]
#     base_dir = os.path.dirname(os.path.abspath(__file__))  # <-- Đây là đường dẫn tuyệt đối
#     create_user_folder_from_defaults(user_id, base_dir=base_dir)
import os
import shutil
import sys

def create_user_folder_from_defaults(user_id, base_dir):
    defaults_dir = os.path.join(base_dir, 'defaults')
    user_folder = os.path.join(base_dir, f'user_{user_id}')
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)
        print(f"[INFO] Created user folder: {user_folder}")
    else:
        print(f"[INFO] User folder already exists: {user_folder}")
        return  # <<< Chỉ chạy khi chưa có folder
    if not os.path.exists(defaults_dir):
        print(f"[ERROR] Defaults folder not found: {defaults_dir}")
        return
    for item in os.listdir(defaults_dir):
        s = os.path.join(defaults_dir, item)
        d = os.path.join(user_folder, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
            print(f"[INFO] Copied folder: {item}")
        else:
            shutil.copy2(s, d)
            print(f"[INFO] Copied file: {item}")
    print(f"[SUCCESS] All defaults copied to {user_folder}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python create_user_folder.py [user_id]")
        sys.exit(1)
    user_id = sys.argv[1]
    base_dir = os.path.dirname(os.path.abspath(__file__))
    create_user_folder_from_defaults(user_id, base_dir=base_dir)