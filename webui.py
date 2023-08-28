import os,time,sys,subprocess,pathlib

# 子プロセスを起動
def start_ui():
    global proc
    proc = subprocess.Popen([sys.executable, "chatswitch.py"])

modules_path = os.path.dirname(os.path.realpath(__file__))
restart_flag = pathlib.Path(modules_path + "/restart")
start_ui()

while True:
    if os.path.exists(restart_flag):
        proc.terminate()
        os.remove(restart_flag)
        start_ui()
    time.sleep(1)
