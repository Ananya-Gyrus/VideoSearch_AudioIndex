import subprocess
uuid_ =  subprocess.check_output(
            ["sudo", "dmidecode", "-s", "system-uuid"], stderr=subprocess.DEVNULL
        ).decode().strip()
print(uuid_)