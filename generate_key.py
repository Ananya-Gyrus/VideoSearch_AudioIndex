from cryptography.fernet import Fernet
import base64
import hashlib
 
def get_key_from_password(password: str) -> bytes:
    return base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
 
def encrypt_file(input_file, expiry_date, password):

    key = get_key_from_password(password)
    fernet = Fernet(key)
    with open(input_file, 'rb') as f:
        data = f.read()
        data = fernet.decrypt(data)
        data += b"\n" + expiry_date.encode()
    encrypted = fernet.encrypt(data)
    return encrypted
 
def encrypt_data(data, password):
    key = get_key_from_password(password)
    fernet = Fernet(key)
    encrypted = fernet.encrypt(data)
    return encrypted

def decrypt_file(encrypted_file, password):
    key = get_key_from_password(password)
    fernet = Fernet(key)
 
    with open(encrypted_file, 'rb') as f:
        data = f.read()
    decrypted = fernet.decrypt(data)
 
    return decrypted