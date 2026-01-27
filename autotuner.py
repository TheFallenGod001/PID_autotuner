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
CURR_MOTOR = 3

MAX_POINTS = 300
MAX_PID_VALUES = 10
TIME_PERIOD = 5
TIME_OUT = 5
iterations = 15

startingPoint = 0
endingPoint = 30
delta = 1
min_error_params = [float('inf'), 0, 0, 0]

learning_rate = 0.005
ACCEPTABLE_ERROR = 2.1

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
kp, ki, kd = 0.01, 0.03, 0.001

# =========================================================
# ====================== SPSA ==============================
# =========================================================
alpha = 0.502
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

sC = np.ones(param_count) * 0.02
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
def sigmoid(x, a = 2.1, b = 0.08):
    return 1 / (1 + np.exp(a-b*x))

def unpack_phi(phi_vec):
    global kp, ki, kd
    idx = 0
    if TUNE_P:
        kp = np.exp(phi_vec[idx]); idx += 1
    if TUNE_I:
        ki = np.exp(phi_vec[idx]); idx += 1
    if TUNE_D:
        kd = np.exp(phi_vec[idx])
    if kp > MAX_PID_VALUES : kp = MAX_PID_VALUES
    if kd > MAX_PID_VALUES : kd = MAX_PID_VALUES
    if ki > MAX_PID_VALUES : ki = MAX_PID_VALUES



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
    global curr_iteration, cycle, phi, err_acc, counter, wait, sC, learning_rate

    MAX_STEP = 0.05         
    GRAD_EPS = 1e-2          
    PATIENCE = 5             
    IMPROVEMENT_EPS = 0.02   

    prev_errors = []

    unpack_phi(phi)
    send_command(startingPoint, kp, ki, kd)
    time.sleep(5.0)

    while curr_iteration < iterations:
        # Random SPSA direction
        sDelta[:] = 2 * np.random.randint(0, 2, size=param_count) - 1
        phi_p = phi + sC * sDelta
        unpack_phi(phi_p)
        send_command(endingPoint, kp, ki, kd)
        time.sleep(TIME_PERIOD)

        with shared_lock:
            Jp = err_acc / counter if counter else 0.0
            err_acc = 0.0
            counter = 0

        phi_m = phi - sC * sDelta
        unpack_phi(phi_m)
        send_command(endingPoint, kp, ki, kd)
        time.sleep(TIME_PERIOD)

        with shared_lock:
            Jm = err_acc / counter if counter else 0.0
            err_acc = 0.0
            counter = 0

        g_hat = ((Jp - Jm) / (2.0 * sC)) * sDelta
        grad_norm = np.linalg.norm(g_hat)

        cycle_error = 0.5 * (Jp + Jm)
        prev_errors.append(cycle_error)

        if grad_norm < GRAD_EPS:
            print("Early stop: gradient norm too small (noise-dominated)")
            break

        a_k = learning_rate / ((curr_iteration + 1 + sA) ** alpha)

        step = a_k * g_hat
        step = np.clip(step, -MAX_STEP, MAX_STEP)

        phi -= step

        if cycle_error < ACCEPTABLE_ERROR:
            sC *= 0.95

        unpack_phi(phi)
        send_command(startingPoint, kp, ki, kd)

        if len(prev_errors) >= PATIENCE:
            recent = prev_errors[-PATIENCE:]
            if max(recent) - min(recent) < IMPROVEMENT_EPS:
                print("Early stop: error plateau detected")
                break

        errors.append(cycle_error)

        print(
            f"[Cycle {cycle}] err={cycle_error:.4f} "
            f"|g|={grad_norm:.4e} "
            f"kp={kp:.4f} ki={ki:.4f} kd={kd:.4f}"
        )

        if cycle_error < min_error_params[0]:
            min_error_params[:] = [cycle_error, kp, ki, kd]

        cycle += 1
        curr_iteration += 1
        time.sleep(3.0)

    print("Tuning finished.")
    print(
        f"Minimum params:\n"
        f"error={min_error_params[0]:.4f}\n"
        f"kp={min_error_params[1]:.4f}\n"
        f"ki={min_error_params[2]:.4f}\n"
        f"kd={min_error_params[3]:.4f}"
    )


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
