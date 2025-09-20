# getFeatures720.py

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fftpack import fft
from statsmodels.tsa.ar_model import AutoReg
import time
from datetime import datetime


def generate_combined_features(file_path):
    """
    Processes the all_train.csv file and returns the combined_features.

    Args:
        file_path (str): Path to the all_train.csv file.

    Returns:
        pd.DataFrame: A DataFrame containing combined features.
    """

    # Parameters for sliding window
    sampling_rate = 20  # 20 Hz
    window_size = 52    # 2.56 seconds (52 samples)
    step_size = 26      # 50% overlap

    # Load data
    data_train = pd.read_csv(file_path)

    def moving_average(signal, window_size):
        return np.convolve(signal, np.ones(window_size) / window_size, mode='same')

    def calculate_jerk_signals_and_magnitude(df, x_col, y_col, z_col):
        # Jerk signal (derivative of acceleration)
        jerk_x = np.diff(df[x_col].values, n=1, axis=0)
        jerk_y = np.diff(df[y_col].values, n=1, axis=0)
        jerk_z = np.diff(df[z_col].values, n=1, axis=0)

        # Add a padding to match the length of the original data
        jerk_x = np.pad(jerk_x, (1, 0), 'constant', constant_values=(0, 0))
        jerk_y = np.pad(jerk_y, (1, 0), 'constant', constant_values=(0, 0))
        jerk_z = np.pad(jerk_z, (1, 0), 'constant', constant_values=(0, 0))

        # Smoothing the jerk signals
        jerk_x = moving_average(jerk_x, window_size=5)
        jerk_y = moving_average(jerk_y, window_size=5)
        jerk_z = moving_average(jerk_z, window_size=5)

        # Magnitude calculation (Euclidean norm)
        magnitude = np.sqrt(df[x_col] ** 2 + df[y_col] ** 2 + df[z_col] ** 2)
        jerk_magnitude = np.sqrt(jerk_x ** 2 + jerk_y ** 2 + jerk_z ** 2)

        # Add to the dataframe
        df[f'{x_col}_Jerk'] = jerk_x
        df[f'{y_col}_Jerk'] = jerk_y
        df[f'{z_col}_Jerk'] = jerk_z
        df[f'{x_col.split("_")[0]}_Mag'] = magnitude
        df[f'{x_col.split("_")[0]}_Jerk_Mag'] = jerk_magnitude

        # Rename columns
        if (x_col, y_col, z_col) == ('DMUAccelX', 'DMUAccelY', 'DMUAccelZ'):
            df.rename(columns={f'{x_col.split("_")[0]}_Mag': 'DMUAccel_Mag',
                               f'{x_col.split("_")[0]}_Jerk_Mag': 'DMUAccel_Jerk_Mag'}, inplace=True)
        elif (x_col, y_col, z_col) == ('DMRoll', 'DMPitch', 'DMYaw'):
            df.rename(columns={f'{x_col.split("_")[0]}_Mag': 'DMOrientation_Mag',
                               f'{x_col.split("_")[0]}_Jerk_Mag': 'DMOrientation_Jerk_Mag'}, inplace=True)
        elif (x_col, y_col, z_col) == ('DMRotX', 'DMRotY', 'DMRotZ'):
            df.rename(columns={f'{x_col.split("_")[0]}_Mag': 'DMRot_Mag',
                               f'{x_col.split("_")[0]}_Jerk_Mag': 'DMRot_Jerk_Mag'}, inplace=True)

        elif (x_col, y_col, z_col) == ('DMGrvX', 'DMGrvY', 'DMGrvZ'):
            df.rename(columns={f'{x_col.split("_")[0]}_Mag': 'DMGrv_Mag',
                               f'{x_col.split("_")[0]}_Jerk_Mag': 'DMGrv_Jerk_Mag'}, inplace=True)
    
        elif (x_col, y_col, z_col) == ('DMQuatX', 'DMQuatY', 'DMQuatZ'):
            df.rename(columns={f'{x_col.split("_")[0]}_Mag': 'DMQuat_Mag',
                               f'{x_col.split("_")[0]}_Jerk_Mag': 'DMQuat_Jerk_Mag'}, inplace=True)

        return df

    def extract_features(window, signal_name):
        mean_val = np.mean(window)
        std_dev = np.std(window)
        mad_val = np.median(np.abs(window - np.median(window)))
        min_val = np.min(window)
        max_val = np.max(window)
        range_val = max_val - min_val
        sma_val = np.sum(np.abs(window))
        energy_val = np.sum(window ** 2) / len(window)
        iqr_val = np.percentile(window, 75) - np.percentile(window, 25)
        ar_model = AutoReg(window, lags=4).fit()
        ar_coeffs = ar_model.params
        freqs = fft(window)
        freqs = np.abs(freqs[:len(freqs) // 2])
        power = freqs ** 2
        max_ind = np.argmax(power)  # dominant_frequency
        total_power = np.sum(power)
        mean_freq = np.sum(freqs * power) / total_power if total_power > 0 else 0
        skewness_val = skew(freqs)
        kurtosis_val = kurtosis(freqs)

        features = {
            f'{signal_name}_mean': mean_val,
            f'{signal_name}_std_dev': std_dev,
            f'{signal_name}_mad': mad_val,
            f'{signal_name}_min': min_val,
            f'{signal_name}_max': max_val,
            f'{signal_name}_range': range_val,
            f'{signal_name}_sma': sma_val,
            f'{signal_name}_energy': energy_val,
            f'{signal_name}_iqr': iqr_val,
            f'{signal_name}_arCoeff1': ar_coeffs[1] if len(ar_coeffs) > 1 else 0,
            f'{signal_name}_arCoeff2': ar_coeffs[2] if len(ar_coeffs) > 2 else 0,
            f'{signal_name}_arCoeff3': ar_coeffs[3] if len(ar_coeffs) > 3 else 0,
            f'{signal_name}_arCoeff4': ar_coeffs[4] if len(ar_coeffs) > 4 else 0,
            f'{signal_name}_power': total_power,
            f'{signal_name}_maxInd': max_ind,
            f'{signal_name}_meanFreq': mean_freq,
            f'{signal_name}_skewness': skewness_val,
            f'{signal_name}_kurtosis': kurtosis_val,
        }

        return features

    def sliding_window_analysis_optimized(df, signals):
        all_features = []
        for name, group in df.groupby(['UAMSWH', 'SessionID']):
            indices = group.index
            sliding_windows = {
                signal_name: np.lib.stride_tricks.sliding_window_view(signal[indices], window_size)[::step_size]
                for signal_name, signal in signals.items()
            }
            num_windows = len(next(iter(sliding_windows.values())))
            for window_idx in range(num_windows):
                combined_features = {}
                for signal_name, windows in sliding_windows.items():
                    window = windows[window_idx]
                    features = extract_features(window, signal_name)
                    combined_features.update(features)
                combined_features['UAMSWH'] = group['UAMSWH'].iloc[0]
                combined_features['SessionID'] = group['SessionID'].iloc[0]
                combined_features['MoveType'] = group['MoveType'].iloc[0]
                all_features.append(combined_features)
        return pd.DataFrame(all_features)

    # Process the data
    data_train = calculate_jerk_signals_and_magnitude(data_train, 'DMUAccelX', 'DMUAccelY', 'DMUAccelZ')
    data_train = calculate_jerk_signals_and_magnitude(data_train, 'DMRoll', 'DMPitch', 'DMYaw')
    data_train = calculate_jerk_signals_and_magnitude(data_train, 'DMRotX', 'DMRotY', 'DMRotZ')
    data_train = calculate_jerk_signals_and_magnitude(data_train, 'DMGrvX', 'DMGrvY', 'DMGrvZ')
    data_train = calculate_jerk_signals_and_magnitude(data_train, 'DMQuatX', 'DMQuatY', 'DMQuatZ')

    # Define the signals dictionary
    signals = {
        'DMUAccel_X': data_train['DMUAccelX'].values,
        'DMUAccel_Y': data_train['DMUAccelY'].values,
        'DMUAccel_Z': data_train['DMUAccelZ'].values,
        'DMUAccel_X_Jerk': data_train['DMUAccelX_Jerk'].values,
        'DMUAccel_Y_Jerk': data_train['DMUAccelY_Jerk'].values,
        'DMUAccel_Z_Jerk': data_train['DMUAccelZ_Jerk'].values,
        'DMUAccel_Mag': data_train['DMUAccel_Mag'].values,
        'DMUAccel_Jerk_Mag': data_train['DMUAccel_Jerk_Mag'].values,
        ###########
        'DMRoll': data_train['DMRoll'].values,
        'DMPitch': data_train['DMPitch'].values,
        'DMYaw': data_train['DMYaw'].values,
        'DMRoll_Jerk': data_train['DMRoll_Jerk'].values,
        'DMPitch_Jerk': data_train['DMPitch_Jerk'].values,
        'DMYaw_Jerk': data_train['DMYaw_Jerk'].values,
        'DMOrientation_Mag': data_train['DMOrientation_Mag'].values,
        'DMOrientation_Jerk_Mag': data_train['DMOrientation_Jerk_Mag'].values,
        ###########
        'DMRotX': data_train['DMRotX'].values,
        'DMRotY': data_train['DMRotY'].values,
        'DMRotZ': data_train['DMRotZ'].values,
        'DMRotX_Jerk': data_train['DMRotX_Jerk'].values,
        'DMRotY_Jerk': data_train['DMRotY_Jerk'].values,
        'DMRotZ_Jerk': data_train['DMRotZ_Jerk'].values,
        'DMRot_Mag': data_train['DMRot_Mag'].values,
        'DMRot_Jerk_Mag': data_train['DMRot_Jerk_Mag'].values,
        ###########
        'DMGrvX': data_train['DMGrvX'].values,
        'DMGrvY': data_train['DMGrvY'].values,
        'DMGrvZ': data_train['DMGrvZ'].values,
        'DMGrvX_Jerk': data_train['DMGrvX_Jerk'].values,
        'DMGrvY_Jerk': data_train['DMGrvY_Jerk'].values,
        'DMGrvZ_Jerk': data_train['DMGrvZ_Jerk'].values,
        'DMGrv_Mag': data_train['DMGrv_Mag'].values,
        'DMGrv_Jerk_Mag': data_train['DMGrv_Jerk_Mag'].values,
        ###########
        'DMQuatX': data_train['DMQuatX'].values,
        'DMQuatY': data_train['DMQuatY'].values,
        'DMQuatZ': data_train['DMQuatZ'].values,
        'DMQuatX_Jerk': data_train['DMQuatX_Jerk'].values,
        'DMQuatY_Jerk': data_train['DMQuatY_Jerk'].values,
        'DMQuatZ_Jerk': data_train['DMQuatZ_Jerk'].values,
        'DMQuat_Mag': data_train['DMQuat_Mag'].values,
        'DMQuat_Jerk_Mag': data_train['DMQuat_Jerk_Mag'].values
    }

    start_time = time.time()
    formatted_time = datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')

    print("===================================")
    print("THE PROCESS START AT "+ formatted_time)
    print("...-------------------...")
    
    combined_features = sliding_window_analysis_optimized(data_train, signals)
    
    # Calculate the total time taken
    total_time = time.time() - start_time
    
    print("===================================")
    print('"THE PROCESS TOOK :"', f"{total_time:.{2}f}", 'seconds', '  |', f"{total_time/60:.{2}f}", 'minutes')

    return combined_features
