import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d


def calculate_delay(t, sent_signal, measured_signal, signal_name):
    # 1. Create a perfectly uniform time grid based on median sampling time
    dt = np.median(np.diff(t))
    t_uniform = np.arange(t[0], t[-1], dt)

    # 2. Interpolate signals onto the uniform grid
    f_sent = interp1d(t, sent_signal, kind='linear', fill_value="extrapolate")
    f_meas = interp1d(t, measured_signal, kind='linear', fill_value="extrapolate")

    sent_uniform = f_sent(t_uniform)
    meas_uniform = f_meas(t_uniform)

    # 3. Normalize signals (zero mean, unit variance) to prevent amplitude bias
    sent_norm = (sent_uniform - np.mean(sent_uniform)) / (np.std(sent_uniform) + 1e-9)
    meas_norm = (meas_uniform - np.mean(meas_uniform)) / (np.std(meas_uniform) + 1e-9)

    # 4. Compute Cross-Correlation
    # correlate computes the similarity of the two signals at all possible time lags
    corr = signal.correlate(meas_norm, sent_norm, mode='full')
    lags = signal.correlation_lags(len(meas_norm), len(sent_norm), mode='full')

    # 5. Find the lag that maximizes correlation
    # We restrict to positive lags (assuming causality: measurement happens AFTER prediction)
    positive_lags_mask = lags >= 0
    lags_pos = lags[positive_lags_mask]
    corr_pos = corr[positive_lags_mask]

    best_lag_idx = np.argmax(corr_pos)
    best_lag = lags_pos[best_lag_idx]

    # Calculate delay in seconds
    delay_seconds = best_lag * dt

    # --- Plotting to visually verify alignment ---
    plt.figure(figsize=(12, 6))

    # Subplot 1: Original vs Aligned Signals
    plt.subplot(2, 1, 1)
    plt.plot(t_uniform, sent_norm, label='Sent (x_pred)', color='blue', alpha=0.7)
    plt.plot(t_uniform, meas_norm, label='Measured (x_meas)', color='red', alpha=0.7)

    # Shift the measured signal back by the calculated delay to show the match
    t_shifted = t_uniform - delay_seconds
    plt.plot(t_shifted, meas_norm, label=f'Measured shifted by -{delay_seconds:.3f}s',
             color='green', linestyle='dashed', linewidth=2)

    plt.title(f'Signal Alignment: {signal_name} (Calculated Delay: {delay_seconds:.3f} s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Cross-Correlation curve
    plt.subplot(2, 1, 2)
    time_lags = lags_pos * dt
    plt.plot(time_lags, corr_pos, color='purple')
    plt.axvline(x=delay_seconds, color='black', linestyle='--', label=f'Max Correlation at {delay_seconds:.3f}s')
    plt.xlim(0, max(1.0, delay_seconds * 3))  # Zoom in on reasonable delay range (0 to 1s)
    plt.title('Cross-Correlation vs Time Lag')
    plt.xlabel('Time Lag (s)')
    plt.ylabel('Correlation Coefficient')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return delay_seconds


def main():
    # Adjust this path to where your CSV actually lives
    csv_path = '../data/offline_viz/csv/step_log.csv'

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Could not find {csv_path}. Please check the path.")
        return

    # Extract time
    t = df['t_rel'].values

    # NOTE: You may need to change these column names to match exactly
    # what is in your step_log.csv header!
    # For example, they might be 'x_pred_velocity_v_x' or 'x_pred_v_x' depending on your state_names.

    # --- VELOCITY DELAY ---
    try:
        # Replace with your actual column names for velocity
        v_pred = df['x_pred_velocity_v_x'].values
        v_meas = df['x_meas_velocity_v_x'].values

        v_delay = calculate_delay(t, v_pred, v_meas, "Velocity (vx)")
        print(f"-> Estimated Velocity Delay: {v_delay:.4f} seconds")
    except KeyError as e:
        print(f"Column not found for Velocity: {e}. Check your CSV headers.")

    # --- STEERING DELAY ---
    try:
        # Replace with your actual column names for steering
        steer_pred = df['x_pred_steering_angle_delta'].values
        steer_meas = df['x_meas_steering_angle_delta'].values

        steer_delay = calculate_delay(t, steer_pred, steer_meas, "Steering Angle")
        print(f"-> Estimated Steering Delay: {steer_delay:.4f} seconds")
    except KeyError as e:
        print(f"Column not found for Steering: {e}. Check your CSV headers.")


if __name__ == "__main__":
    main()
