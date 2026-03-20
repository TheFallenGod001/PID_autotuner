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
PORT = "COM11"
BAUD = 115200
CURR_MOTOR = 2

MAX_POINTS = 300
TIME_PERIOD = 2
TIME_OUT = 5
iterations = 20
intermediate_iterations = 2

startingPoint = 0
endingPoint = 30
delta = 1
min_error_params = [float('inf'), 0, 0, 0]

learning_rate = 0.009
ACCEPTABLE_ERROR = 30.0
P_MIN = 0.001
P_MAX = 600
I_MIN = 0.001
I_MAX = 300
D_MIN = 0.00001
D_MAX = 20

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
kp, ki, kd = 5.34793630178868, 90.2035143774313, 0.0010263981928690996

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
osc_err_acc = 0.0
p_rpm = 0
counter = 0
cycle = 0
curr_iteration = 0
accumulate  = True

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
def unpack_phi(phi_vec):
    global kp, ki, kd
    idx = 0
    if TUNE_P:
        kp = np.exp(phi_vec[idx]); idx += 1
    if TUNE_I:
        ki = np.exp(phi_vec[idx]); idx += 1
    if TUNE_D:
        kd = np.exp(phi_vec[idx])
    kp = np.clip(kp, P_MIN, P_MAX)
    ki = np.clip(ki, I_MIN, I_MAX)
    kd = np.clip(kd, D_MIN, D_MAX)



def send_command(target_rpm, kp_val=None, ki_val=None, kd_val=None):
    cmd = f"0 {int (target_rpm)} "
    if kp_val is not None: cmd += f"{kp_val:.6f} "
    if ki_val is not None: cmd += f"{ki_val:.6f} "
    if kd_val is not None: cmd += f"{kd_val:.6f}"
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
    global act_rpm, err_acc, counter, osc_err_acc, p_rpm

    while True:

        line = ser.readline().decode(errors='ignore').strip()
        if not line:
            continue

        parts = line.split(",")
        if len(parts) == 2:
            with shared_lock:
                if (accumulate):
                    act_rpm = safe_float(parts[0])
                    exp_rpm = safe_float(parts[1])
                    err_acc += (exp_rpm - act_rpm) ** 2
                    counter += 1
                    osc_err_acc += (act_rpm - p_rpm) ** 2
                    p_rpm = act_rpm
                    data_buffer.append((act_rpm, exp_rpm))

threading.Thread(target=read_serial, daemon=True).start()

# =========================================================
# ==================== TUNER ==============================
# =========================================================
def tuner():
    global curr_iteration, cycle, phi, err_acc, counter, sC, accumulate 

    unpack_phi(phi)
    send_command(startingPoint, kp, ki, kd)
    time.sleep(1.5)

    while curr_iteration < iterations:
        Jp = []
        Jm = []
        for i in range(intermediate_iterations):
            sDelta[:] = 2 * np.random.randint(0, 2, size=param_count) - 1

            # ====================== J+ ======================
            phi_p = phi + sC * sDelta
            unpack_phi(phi_p)
            send_command(endingPoint, kp, ki, kd)

            if EVAL_MODE == "TIME":
                with shared_lock:
                    err_acc = 0
                    counter = 0
                    osc_err_acc = 0
                time.sleep(TIME_PERIOD)
            elif EVAL_MODE == "DELTA":
                wait_until_settled(endingPoint, delta, TIME_OUT)
                with shared_lock:
                    err_acc = 0.0
                    counter = 0
                    osc_err_acc = 0
                time.sleep(DELTA_SAMPLE_TIME)

            with shared_lock:
                accumulate  = False
                Jp.append((err_acc / counter if counter else 0.0) + (osc_err_acc / counter if counter else 0.0))
                err_acc = 0.0
                counter = 0
                osc_err_acc = 0
                accumulate  = True
            send_command(startingPoint, kp, ki, kd)
            time.sleep(1.5)

            # ====================== J- ======================
            phi_m = phi - sC * sDelta
            unpack_phi(phi_m)
            send_command(endingPoint, kp, ki, kd)

            if EVAL_MODE == "TIME":
                with shared_lock:
                    err_acc = 0
                    counter = 0
                    osc_err_acc = 0
                time.sleep(TIME_PERIOD)
            elif EVAL_MODE == "DELTA":
                wait_until_settled(endingPoint, delta, TIME_OUT)
                with shared_lock:
                    err_acc = 0.0
                    counter = 0
                    osc_err_acc = 0
                time.sleep(DELTA_SAMPLE_TIME)

            with shared_lock:
                accumulate  = False
                Jm.append((err_acc / counter if counter else 0.0) + (osc_err_acc / counter if counter else 0.0))
                err_acc = 0.0
                counter = 0
                osc_err_acc = 0
                accumulate  = True
            send_command(startingPoint, kp, ki, kd)
            time.sleep(1.5)
        # ====================== SPSA UPDATE ======================
        g_hat = ((np.mean(Jp) - np.mean(Jm)) / (2.0 * sC)) * sDelta
        g_hat = np.clip(g_hat, -250, 250)
        a_k = learning_rate / ((curr_iteration + 1 + sA) ** alpha)

        cycle_error = 0.5 * (np.mean(Jp) + np.mean(Jm))
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
            f"a_eff={a_eff:.6e} kp={kp:.8f} ki={ki:.8f} kd={kd:.8f}"
        )
        if (cycle_error < min_error_params[0]):
            min_error_params[0] = cycle_error
            min_error_params[1] = kp
            min_error_params[2] = ki
            min_error_params[3] = kd

        cycle += 1
        curr_iteration += 1
        time.sleep(1.5)

    print("Tuning finished.")
    print(f"Minimum params obtained from tune: minimum error - {min_error_params[0]}\nkp - {min_error_params[1]}\nki - {min_error_params[2]}\nkd - {min_error_params[3]}")

threading.Thread(target=tuner, daemon=True).start()

# =========================================================
# ==================== PLOTTING ===========================
fig, (ax_rpm, ax_cost) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=False,
    gridspec_kw={"height_ratios": [3, 1]}
)

# ---------- RPM plot ----------
x_rpm = np.arange(MAX_POINTS)

line_act, = ax_rpm.plot(x_rpm, np.zeros(MAX_POINTS), label="Actual RPM")
line_exp, = ax_rpm.plot(x_rpm, np.zeros(MAX_POINTS), label="Target RPM")

ax_rpm.set_ylabel("RPM")
ax_rpm.set_ylim(-120, 120)
ax_rpm.legend()
ax_rpm.grid(True)

# ---------- Cost plot ----------
line_cost, = ax_cost.plot([], [], 'r-o', markersize=4)
ax_cost.set_ylabel("Cost (MSE)")
ax_cost.set_xlabel("SPSA Cycle")
ax_cost.grid(True)

def update(frame):
    # -------- RPM data --------
    with shared_lock:
        db = list(data_buffer)
        cost_data = errors.copy()

    if db:
        a = np.array([d[0] for d in db])
        e = np.array([d[1] for d in db])

        a = np.pad(a, (MAX_POINTS - len(a), 0))[-MAX_POINTS:]
        e = np.pad(e, (MAX_POINTS - len(e), 0))[-MAX_POINTS:]

        line_act.set_ydata(a)
        line_exp.set_ydata(e)

    # -------- Cost data --------
    if cost_data:
        x_cost = np.arange(len(cost_data))
        line_cost.set_data(x_cost, cost_data)
        ax_cost.set_xlim(0, max(5, len(cost_data)))
        ax_cost.set_ylim(0, max(cost_data) * 1.1)

    return line_act, line_exp, line_cost

ani = animation.FuncAnimation(
    fig,
    update,          # <-- THIS is where update is “called”
    interval=50,     # ms between frames
    blit=True
)

plt.show()

