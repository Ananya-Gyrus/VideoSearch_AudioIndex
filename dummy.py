# Python shell or a script
from generate_key import encrypt_data

dummy_info = b"dummy_hardware_info"
password = "Giuy439rfhcb$uix-b312"

encrypted = encrypt_data(dummy_info, password)

with open("work_dir/client_hardware_info.txt", "wb") as f:
    f.write(encrypted)
