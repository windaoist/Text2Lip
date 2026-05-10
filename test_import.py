import os
import sys

# Simulation of what text_driven_prototype.py does
PROJECT_ROOT = os.getcwd()
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print(f"sys.path[0]: {sys.path[0]}")

try:
    from echomimic_backend import EchoMimicBackend
    print("[+] Successfully imported EchoMimicBackend")
except ImportError as e:
    print(f"[-] ImportError: {e}")
except Exception as e:
    print(f"[-] Other error: {e}")
    import traceback
    traceback.print_exc()
