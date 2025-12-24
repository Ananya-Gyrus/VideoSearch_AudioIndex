from cryptography.fernet import Fernet
import base64
import hashlib
 
def get_key_from_password(password: str) -> bytes:
    # Derive a 32-byte key from the password
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())

def encrypt_file(input_file, expiry_date, hourly_credits,recent_date, password):
    key = Fernet.generate_key()
    # print("generated key:", key)
    key = get_key_from_password(password)
    fernet = Fernet(key)
    # print("ferney:", fernet)
    with open(input_file, 'rb') as f:
        data = f.read()
        data = fernet.decrypt(data)
        print(data)
        data += b"\n" + expiry_date.encode()
        data += b"\n" + str(hourly_credits).encode()
        data += b"\n" + recent_date.encode()
    encrypted = fernet.encrypt(data)
    return encrypted
 
print(encrypt_file("work_dir/client_hardware_info.txt", "2026-11-10T00:00:00", 1000,"2025-05-10T00:00:00", "Giuy439rfhcb$uix-b312").decode())
with open("work_dir/licence_key.txt", "w") as f:
    f.write(encrypt_file("work_dir/client_hardware_info.txt", "2026-11-10T00:00:00", 1000,"2025-05-10T00:00:00", "Giuy439rfhcb$uix-b312").decode())