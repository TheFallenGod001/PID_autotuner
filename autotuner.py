import serial
import threading
import time
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =========================================================
# ======================= CONFIG ==========================
# =========================================================
PORT = "COM18"
BAUD = 115200
CURR_MOTOR = 2

MAX_POINTS = 300
MAX_PID_VALUES = 50
TIME_PERIOD = 8
TIME_OUT = 5
iterations = 15

startingPoint = 0
endingPoint = 20
delta = 1
min_error_params = [float('inf'), 0, 0, 0]

learning_rate = 0.001
ACCEPTABLE_ERROR = 0.0

# -------- COST EVALUATION MODE --------
# "TIME"  -> fixed time window (includes transients)
# "DELTA" -> wait for steady state (|rpm-target| <= delta)
EVAL_MODE = "TIME"       
DELTA_SAMPLE_TIME = 2     # seconds of data after settling

# -------- SELECT WHAT TO TUNE --------
TUNE_P = True
TUNE_I = True
TUNE_D = True
# -------- INITIAL GAINS (must be > 0) --------
kp, ki, kd = 0.00489300219772169, 0.02823752341330259, 0.00013409059780944936

# =========================================================
# ====================== SPSA ==============================
# =========================================================
alpha = 0.602
sA = 0.1 * iterations

param_names = []
if TUNE_P: param_names.append("P")
if TUNE_I: param_names.append("I")
if TUNE_D: param_names.append("D")

param_count = len(param_names)
assert param_count > 0, "At least one parameter must be tuned"

# Log-space parameters
phi = []
if TUNE_P: phi.append(np.log(kp))
if TUNE_I: phi.append(np.log(ki))
if TUNE_D: phi.append(np.log(kd))
phi = np.array(phi, dtype=float)

sC = np.random.rand(param_count) * 0.01
sDelta = np.zeros(param_count)

# =========================================================
# =================== SHARED STATE ========================
# =========================================================
data_buffer = deque(maxlen=MAX_POINTS)
errors = []

act_rpm = 0.0
err_acc = 0.0
counter = 0
cycle = 0
curr_iteration = 0
wait = False

shared_lock = threading.Lock()

# =========================================================
# ================== SERIAL SETUP =========================
# =========================================================
ser = serial.Serial(PORT, BAUD, timeout=1)
print(f"Opened serial on {PORT} @ {BAUD}")

def safe_float(s, default=0.0):
    try:
        return float(s)
    except:
        return default

# =========================================================
# ================== HELPER FUNCTIONS =====================
# =========================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def unpack_phi(phi_vec):
    global kp, ki, kd
    idx = 0
    if TUNE_P:
        kp = MAX_PID_VALUES*sigmoid(phi_vec[idx]); idx += 1
    if TUNE_I:
        ki = MAX_PID_VALUES*sigmoid(phi_vec[idx]); idx += 1
    if TUNE_D:
        kd = MAX_PID_VALUES*sigmoid(phi_vec[idx])
    if (kp > 20): kp = 20
    if (ki > 20): ki = 20
    if (kd > 20): kd = 20



def send_command(target_rpm, kp_val=None, ki_val=None, kd_val=None):
    cmd = f"B{CURR_MOTOR} {int(target_rpm)} "
    if kp_val is not None: cmd += f"P{kp_val:.4f} "
    if ki_val is not None: cmd += f"I{ki_val:.4f} "
    if kd_val is not None: cmd += f"D{kd_val:.4f};"
    ser.write((cmd + "\n").encode())
    print("Sent:", cmd)

def wait_until_settled(target, delta, timeout):
    start = time.time()
    while True:
        with shared_lock:
            curr = act_rpm
        if abs(curr - target) <= delta:
            return True
        if time.time() - start > timeout:
            return False
        time.sleep(0.05)

# =========================================================
# ================= SERIAL READER =========================
# =========================================================
def read_serial():
    global act_rpm, err_acc, counter

    while True:
        if wait:
            time.sleep(0.01)
            continue

        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) == 2:
            with shared_lock:
                act_rpm = safe_float(parts[0])
                exp_rpm = safe_float(parts[1])
                err_acc += (exp_rpm - act_rpm) ** 2
                counter += 1
                data_buffer.append((act_rpm, exp_rpm))

threading.Thread(target=read_serial, daemon=True).start()

# =========================================================
# ==================== TUNER ==============================
# =========================================================
def tuner():
    global curr_iteration, cycle, phi, err_acc, counter, wait, sC

    unpack_phi(phi)
    send_command(startingPoint, kp, ki, kd)
    time.sleep(5.0)

    while curr_iteration < iterations:
        sDelta[:] = 2 * np.random.randint(0, 2, size=param_count) - 1

        # ====================== J+ ======================
        phi_p = phi + sC * sDelta
        unpack_phi(phi_p)
        send_command(endingPoint, kp, ki, kd)

        if EVAL_MODE == "TIME":
            time.sleep(TIME_PERIOD)
        elif EVAL_MODE == "DELTA":
            wait_until_settled(endingPoint, delta, TIME_OUT)
            with shared_lock:
                err_acc = 0.0
                counter = 0
            time.sleep(DELTA_SAMPLE_TIME)

        with shared_lock:
            Jp = err_acc / counter if counter else 0.0
            err_acc = 0.0
            counter = 0

        # ====================== J- ======================
        phi_m = phi - sC * sDelta
        unpack_phi(phi_m)
        send_command(endingPoint, kp, ki, kd)

        if EVAL_MODE == "TIME":
            time.sleep(TIME_PERIOD)
        elif EVAL_MODE == "DELTA":
            wait_until_settled(endingPoint, delta, TIME_OUT)
            with shared_lock:
                err_acc = 0.0
                counter = 0
            time.sleep(DELTA_SAMPLE_TIME)

        with shared_lock:
            Jm = err_acc / counter if counter else 0.0
            err_acc = 0.0
            counter = 0

        # ====================== SPSA UPDATE ======================
        g_hat = ((Jp - Jm) / (2.0 * sC)) * sDelta
        a_k = learning_rate / ((curr_iteration + 1 + sA) ** alpha)

        cycle_error = 0.5 * (Jp + Jm)
        weight = cycle_error / (cycle_error + ACCEPTABLE_ERROR)
        a_eff = a_k * weight

        phi -= a_eff * g_hat

        if cycle_error < ACCEPTABLE_ERROR:
            sC *= 0.95

        unpack_phi(phi)
        send_command(startingPoint, kp, ki, kd)

        errors.append(cycle_error)
        print(
            f"[Cycle {cycle}] mode={EVAL_MODE} err={cycle_error:.6f} "
            f"a_eff={a_eff:.6e} kp={kp:.4f} ki={ki:.4f} kd={kd:.4f}"
        )
        if (cycle_error < min_error_params[0]):
            min_error_params[0] = cycle_error
            min_error_params[1] = kp
            min_error_params[2] = ki
            min_error_params[3] = kd

        cycle += 1
        curr_iteration += 1
        time.sleep(0.3)

    print("Tuning finished.")
    print(f"Minimum params obtained from tune: minimum error - {min_error_params[0]}\nkp - {min_error_params[1]}\nki - {min_error_params[2]}\nkd - {min_error_params[3]}")

threading.Thread(target=tuner, daemon=True).start()

# =========================================================
# ==================== PLOTTING ===========================
# =========================================================
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(MAX_POINTS)

line_act, = ax.plot(x, np.zeros(MAX_POINTS), label="Actual RPM")
line_exp, = ax.plot(x, np.zeros(MAX_POINTS), label="Expected RPM")
line_err, = ax.plot(x, np.zeros(MAX_POINTS), label="Error")

ax.set_ylim(-120, 120)
ax.legend()
ax.grid(True)

def update(frame):
    with shared_lock:
        db = list(data_buffer)

    if db:
        a = np.array([d[0] for d in db])
        e = np.array([d[1] for d in db])
        a = np.pad(a, (MAX_POINTS - len(a), 0))[-MAX_POINTS:]
        e = np.pad(e, (MAX_POINTS - len(e), 0))[-MAX_POINTS:]
        line_act.set_ydata(a)
        line_exp.set_ydata(e)

    err = np.pad(errors, (MAX_POINTS - len(errors), 0))[-MAX_POINTS:]
    line_err.set_ydata(err)

    return line_act, line_exp, line_err

ani = animation.FuncAnimation(fig, update, interval=50, blit=True)
plt.show()
