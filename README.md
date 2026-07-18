# PID_autotuner

An automatic PID gain tuner for real motor hardware, using **SPSA (Simultaneous Perturbation Stochastic Approximation)** to find good gains through live experiments over a serial connection — no manual trial-and-error, no analytical plant model required.

This is the tuning counterpart to [BDC_Driver_NOVA](../BDC_Driver_NOVA): it talks to that firmware's serial protocol directly, sending RPM setpoints and gain updates and reading back live RPM telemetry to evaluate how good a given set of gains actually is on the real motor.

## Why SPSA

Naively tuning 3 gains (`kp`, `ki`, `kd`) via finite-difference gradient descent requires perturbing each parameter up and down independently — `2 × 3 = 6` real-world trials per gradient estimate. Each "trial" here means physically commanding the motor, waiting for it to respond, and measuring the result — slow, and impossible to parallelize on a single motor.

**SPSA needs only 2 trials per gradient estimate, regardless of how many parameters are being tuned.** It perturbs *all* parameters simultaneously in a random ±1 direction (`sDelta`), evaluates the cost at both perturbed points, and estimates the gradient from that single pair. This is the entire reason SPSA exists as an algorithm, and it's the right tool for exactly this kind of "each evaluation costs real time on real hardware" problem.

## How it works

1. **Log-space parameterization** — gains are tuned as `phi = [log(kp), log(ki), log(kd)]` rather than directly. This guarantees gains can never go negative regardless of how the optimizer perturbs them, and handles gains that naturally span multiple orders of magnitude far better than tuning in linear space.
2. **Each SPSA iteration:**
   - Pick a random ±1 direction for every tuned parameter (`sDelta`)
   - Evaluate the cost at `phi + c·sDelta` (`J+`) by sending those gains to the motor, commanding a step from `startingPoint` to `endingPoint`, and measuring the result
   - Evaluate the cost at `phi - c·sDelta` (`J-`) the same way
   - Estimate the gradient from the difference between the two, and step `phi` in the opposite direction
   - Gradually shrink both the step size (`sC`) and learning rate (`a_k`) over time, following SPSA's standard decaying schedules
3. **Cost function** combines two things:
   - Mean squared error between actual and target RPM
   - An oscillation penalty — squared frame-to-frame RPM deltas — so gains that hit the target but jitter or ring are penalized, not just gains that miss the target
4. **Adaptive effective learning rate** — steps are scaled down as the cost approaches an acceptable threshold (`ACCEPTABLE_ERROR`), so the optimizer takes smaller, more careful steps once it's already performing well, rather than risking overshoot past a good solution. This isn't part of the standard SPSA recipe — it's a refinement added on top of it.
5. Live plots (via `matplotlib.animation`) show real-time RPM tracking and the cost trend across SPSA cycles as tuning runs.

## Configuration

Key parameters at the top of `autotuner.py`:

| Parameter | Purpose |
|---|---|
| `PORT`, `BAUD` | Serial connection to the motor controller |
| `startingPoint`, `endingPoint` | The step input used to evaluate each trial (e.g. 0 → 30 RPM) |
| `iterations`, `intermediate_iterations` | How many SPSA cycles to run, and how many perturbation pairs averaged per cycle |
| `TUNE_P` / `TUNE_I` / `TUNE_D` | Which gains to tune — any subset can be frozen |
| `P_MIN`/`P_MAX`, `I_MIN`/`I_MAX`, `D_MIN`/`D_MAX` | Hard clamps applied after every update, as a safety net against runaway gains |
| `EVAL_MODE` | `"TIME"` — fixed window per trial (includes the transient response); `"DELTA"` — wait for the motor to settle within `delta` of target before sampling |
| `ACCEPTABLE_ERROR` | Threshold used for the adaptive learning-rate damping described above |

The initial gains (`kp, ki, kd = 5.347..., 90.203..., 0.00102...`) are the output of a previous tuning run, the tuner is designed to be re-run and refined incrementally.

## Serial Protocol

Matches [BDC_Driver_NOVA](../BDC_Driver_NOVA)'s expected command format:

```
0 <target_rpm> <kp> <ki> <kd>
```

and reads back telemetry lines in `actual_rpm,expected_rpm` format.

## Running

```bash
pip install pyserial numpy matplotlib
python autotuner.py
```

Requires the motor controller to already be flashed, powered, and connected over serial before starting. The script opens the serial port immediately on import, there's no retry/connection-check step, so make sure the device is present first.

## Known limitations

- `CURR_MOTOR = 2` is defined in the config block but never actually used anywhere in the file. `send_command()` always sends `0` for the left RPM slot and puts the tuning target in the right RPM slot, so as written this can only tune while the left motor is held at zero. This lines up with a matching limitation on the firmware side; `BDC_Driver_NOVA`'s `main.c` currently mirrors one motor's PID gains onto the other unconditionally, so independent per-motor tuning isn't supported end-to-end yet. `CURR_MOTOR` looks like a placeholder for that capability rather than a bug in this file specifically.
- No reconnect/retry logic if the serial port is busy or the device isn't present at startup — `serial.Serial(...)` will just throw.
- `wait_until_settled()` (used in `"DELTA"` eval mode) has a fixed timeout but no explicit handling for a motor that never settles beyond returning `False` and proceeding anyway — worth confirming that's the intended fallback behavior.

## Roadmap

- [ ] Add serial reconnect/retry handling
- [ ] Persist tuning history (gains + cost per cycle) to a file, not just stdout
- [ ] Add a config flag to resume from a previous run's final gains automatically
