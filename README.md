# SPSA PID Auto-Tuner

A Python script that tunes PID gains for a motor by talking to it over serial. It uses SPSA (Simultaneous Perturbation Stochastic Approximation) to search for kp, ki, kd values that minimize tracking error on a step input.

## What it does

Sends step commands to a motor controller, reads back actual vs target RPM, scores how well the motor tracked, and nudges the gains in a direction that lowers the score. There's a live plot showing the RPM trace on top and the cost per cycle on the bottom so you can watch the tune progress.

## Requirements

A motor controller on a serial port that:
* Accepts commands shaped like `<motor_id> <target_rpm> <kp> <ki> <kd>\n`
* Streams back lines of the form `actual_rpm,target_rpm`

Python packages:

```
pip install pyserial numpy matplotlib
```

## Running it

Edit the config block at the top:

```python
PORT = "COM11"
BAUD = 115200
```

Then:

```
python tuner.py
```

A plot window opens and the tuner starts running in a background thread. Each cycle prints the current cost and gains. When it's done, the best gains seen during the run are printed.

## How the tuning works

Each SPSA cycle does roughly this:

1. Pick a random direction in gain-space (each component is +1 or -1).
2. Try gains shifted slightly in that direction, measure the cost. Call this `J+`.
3. Try gains shifted the other way, measure the cost. Call this `J-`.
4. Estimate the gradient from the difference and take a step downhill.

The gains are stored in log space, so updates are multiplicative. That keeps them positive and lets a single learning rate cover gains that differ by orders of magnitude (the integral term is often hundreds while the derivative term is in the thousandths).

The cost itself is mean-squared tracking error plus an oscillation penalty (squared change in RPM between consecutive samples). The oscillation term matters because plain MSE doesn't punish ringing nearly enough.

## Two evaluation modes

Set `EVAL_MODE` to one of:

* `"TIME"` runs a fixed `TIME_PERIOD` seconds after the step command. The transient is part of the score, so slow rise time and overshoot both get punished.
* `"DELTA"` waits until the motor is within `delta` RPM of the target, then samples for `DELTA_SAMPLE_TIME` seconds. This measures steady-state behavior only.

Use TIME if you care about how fast it gets there. Use DELTA if you only care about how cleanly it holds the setpoint.

## Config you'll probably want to touch

* `iterations` — number of SPSA cycles to run
* `intermediate_iterations` — how many J+/J- pairs to average per cycle. Higher means a less noisy gradient estimate but a slower tune.
* `startingPoint` and `endingPoint` — the step you're tuning against (e.g. 0 to 30 RPM)
* `learning_rate` — base SPSA step size
* `ACCEPTABLE_ERROR` — once cycle cost drops below this, the perturbation size shrinks and the effective step size scales down. Helps the tune settle instead of bouncing around the optimum.
* `TUNE_P`, `TUNE_I`, `TUNE_D` — set any of these to `False` to freeze that gain at its initial value
* `kp`, `ki`, `kd` — your starting guess. They have to be positive (we're in log space).

The clip limits (`P_MIN`, `P_MAX`, etc.) exist so a bad gradient estimate can't fling the gains somewhere insane.

## A few notes

The serial reader and the tuner share an `accumulate` flag protected by a lock. The reader stops adding to the error accumulator while the tuner is reading it out. There's still a small race window in there but it hasn't caused problems in practice.

The plot animation updates every 50 ms. If your machine struggles with that, bump `interval` in the `FuncAnimation` call.

The best gains seen during the run are tracked in `min_error_params` and printed at the end. The final `phi` is not necessarily the best (SPSA doesn't monotonically improve), so prefer the printed minimum.

## Output format

Each cycle logs something like:

```
[Cycle 4] mode=TIME err=12.348721 a_eff=3.42e-04 kp=6.12834000 ki=82.41008100 kd=0.00128400
```

`err` is the average of J+ and J- for that cycle. `a_eff` is the effective step size (learning rate scaled by how far from `ACCEPTABLE_ERROR` you are).
