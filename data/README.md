# WebXR Data Collection Files

This directory contains comprehensive data files collected from the WebXR IMU & Acoustic Data Collector application.

## File Types

### XR Pose Data Files (`xr_pose_data_*.csv`)
- **Format**: CSV with Unix timestamps and viewer head pose data
- **Columns**: UnixTime(ms), PosX, PosY, PosZ, OrientX, OrientY, OrientZ, OrientW
- **Sample Rate**: Target 800Hz
- **Description**: Primary head position and orientation from WebXR viewer pose
- **Units**: 
  - Position (PosX, PosY, PosZ): meters
  - Orientation (OrientX, OrientY, OrientZ, OrientW): quaternion (normalized)

### XR Eye View Data Files (`xr_eye_views_*.csv`)
- **Format**: CSV with eye-specific transform data
- **Columns**: UnixTime(ms), ViewIndex, Eye, EyePosX, EyePosY, EyePosZ, EyeOrientX, EyeOrientY, EyeOrientZ, EyeOrientW
- **Sample Rate**: Target 800Hz per eye (1600Hz total for stereo)
- **Description**: Individual eye positions and orientations for stereo rendering
- **Eye Values**: "left", "right", or "none"
- **Units**: 
  - Position: meters relative to head
  - Orientation: quaternion (normalized)

### XR Projection Matrix Data Files (`xr_projection_matrices_*.csv`)
- **Format**: CSV with 4x4 projection matrices for each eye
- **Columns**: UnixTime(ms), ViewIndex, Eye, M00-M33 (16 matrix elements)
- **Sample Rate**: Target 800Hz per eye
- **Description**: Projection matrices used for stereo rendering calculations
- **Usage**: Essential for accurate 3D reconstruction and calibration

### XR Path Log Files (`xr_path_log_*.csv`)
- **Format**: CSV with comprehensive XR tracking metadata and pose data
- **Session Metadata Header**: `# Session Metadata: sessionId,referenceSpaceType,units,handedness,upAxis,forwardAxis,headsetHeightM`
- **Columns**: session_id, type, UnixTime(ms), PosX, PosY, PosZ, OrientX, OrientY, OrientZ, OrientW, trackingState
- **Sample Rate**: Target 800Hz
- **Description**: Complete XR path reconstruction data with tracking quality indicators
- **Reference Space**: Prefers `local-floor`, falls back to `local` then `viewer`
- **Coordinate System**: Right-handed, +Y up, -Z forward (typical WebXR conventions)
- **Tracking States**: 
  - `normal`: High-quality tracking with good pose data
  - `limited`: Tracking issues detected (sudden jumps, invalid quaternions)
  - `lost`: No valid pose data available
- **Units**: Meters for position, normalized quaternions for orientation
- **Use Cases**: Drift-free trajectory reconstruction, floor plane projection, yaw extraction

### Derived IMU Data Files (`imu_derived_motion_*.csv`)
- **Format**: CSV with calculated motion data derived from XR pose
- **Columns**: UnixTime(ms), Ax, Ay, Az, Gx, Gy, Gz, QuatW, QuatX, QuatY, QuatZ
- **Sample Rate**: Target 800Hz
- **Description**: Calculated acceleration and angular velocity from pose differences
- **Units**: 
  - Acceleration (Ax, Ay, Az): m/s²
  - Gyroscope (Gx, Gy, Gz): rad/s
  - Quaternion (QuatW, QuatX, QuatY, QuatZ): normalized

### Raw Audio Files (`raw_audio_stereo_*.wav`)
- **Format**: Uncompressed WAV (PCM 16-bit)
- **Sample Rate**: 48kHz
- **Channels**: 2 (Stereo) - interleaved Left/Right
- **Capture Method**: Raw PCM via AudioWorklet/ScriptProcessor
- **Quality**: Lossless, full-resolution stereo audio data
- **Description**: Direct microphone samples without compression
- **Note**: If mono input is detected, left channel is duplicated to right channel

### Chirp Timing Log Files (`raw_audio_stereo_*.txt`)
- **Format**: Plain text, one event per line
- **Naming**: Same base name as corresponding WAV file with .txt extension
- **Content**: Recording start time and chirp playback timestamps
- **Time Format**: Unix timestamp in seconds (decimal)
- **Example Content**:
  ```
  Stereo Recording Started at 1755640584.413
  FMCW Linear 16-20kHz Playback Started at 1755640587.193
  FMCW Linear 16-20kHz Playback Started at 1755640589.170
  ```

### Session Metadata (`session_metadata_*.json`)
- **Format**: JSON
- **Contents**: Session duration, sample counts, device info, exported files list

## Data Processing

### Processing Raw Audio Files
Raw audio files are already in WAV format and ready for analysis:

```python
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal

# Load raw stereo WAV file
sample_rate, audio_data = wav.read('raw_audio_stereo_2024-01-15_10-30-45_EST.wav')
print(f"Sample rate: {sample_rate}Hz, Channels: {audio_data.shape[1] if len(audio_data.shape) > 1 else 1}")
print(f"Duration: {len(audio_data)/sample_rate:.2f}s")

# Convert to float32 for analysis
if len(audio_data.shape) > 1:
    # Stereo audio - separate channels
    left_channel = audio_data[:, 0].astype(np.float32) / 32768.0
    right_channel = audio_data[:, 1].astype(np.float32) / 32768.0
    audio_float = left_channel  # Use left channel for analysis, or combine both
else:
    # Mono audio (fallback)
    audio_float = audio_data.astype(np.float32) / 32768.0

# Analyze chirp responses
def find_chirp_responses(audio, fs, chirp_freq_range=(16000, 20000)):
    # Design bandpass filter for chirp frequency range
    nyq = fs // 2
    low = chirp_freq_range[0] / nyq
    high = chirp_freq_range[1] / nyq
    b, a = signal.butter(6, [low, high], btype='band')
    
    # Filter audio
    filtered = signal.filtfilt(b, a, audio)
    
    # Find peaks (potential chirp responses)
    peaks, _ = signal.find_peaks(np.abs(filtered), height=0.1, distance=fs//10)
    
    return peaks, filtered

# Detect chirp responses
peaks, filtered_audio = find_chirp_responses(audio_float, sample_rate)
print(f"Found {len(peaks)} potential chirp responses")

# Spectral analysis
freqs, times, spectrogram = signal.spectrogram(audio_float, sample_rate, nperseg=1024)

# Load and process chirp timing log
timing_file = 'raw_audio_stereo_2024-01-15_10-30-45_EST.txt'
with open(timing_file, 'r') as f:
    timing_lines = f.readlines()

# Parse timing events
recording_start = None
chirp_times = []

for line in timing_lines:
    line = line.strip()
    if 'Recording Started at' in line:
        recording_start = float(line.split('at ')[1])
    elif 'Playback Started at' in line:
        chirp_times.append(float(line.split('at ')[1]))

print(f"Recording started at: {recording_start}")
print(f"Chirp playback times: {chirp_times}")

# Calculate relative timing (seconds from recording start)
if recording_start:
    relative_chirp_times = [(t - recording_start) for t in chirp_times]
    print(f"Relative chirp times: {relative_chirp_times}")
    
    # Convert to sample indices for audio analysis
    chirp_sample_indices = [int(t * sample_rate) for t in relative_chirp_times]
    print(f"Chirp sample indices: {chirp_sample_indices}")
    
    # Analyze audio around chirp times
    for i, sample_idx in enumerate(chirp_sample_indices):
        if 0 <= sample_idx < len(audio_float):
            # Extract 100ms window around chirp
            window_samples = int(0.1 * sample_rate)  # 100ms
            start_idx = max(0, sample_idx - window_samples//2)
            end_idx = min(len(audio_float), sample_idx + window_samples//2)
            chirp_window = audio_float[start_idx:end_idx]
            
            # Analyze this window
            chirp_rms = np.sqrt(np.mean(chirp_window**2))
            print(f"Chirp {i+1} RMS level: {chirp_rms:.4f}")
```

### Processing XR Data
All XR data can be processed with standard CSV libraries:

```python
import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation

# 1. Load XR Pose Data (Head tracking)
pose_df = pd.read_csv('xr_pose_data_2024-01-15T10-30-45.csv')
pose_df['timestamp'] = pd.to_datetime(pose_df['UnixTime(ms)'], unit='ms')

# Extract head position and orientation
head_position = pose_df[['PosX', 'PosY', 'PosZ']].values
head_quaternion = pose_df[['OrientW', 'OrientX', 'OrientY', 'OrientZ']].values

# Convert quaternions to Euler angles
head_rotation = Rotation.from_quat(head_quaternion[:, [1,2,3,0]])  # scipy uses xyzw order
euler_angles = head_rotation.as_euler('xyz', degrees=True)

# 2. Load Eye View Data (Stereo tracking)
eye_df = pd.read_csv('xr_eye_views_2024-01-15T10-30-45.csv')
left_eye = eye_df[eye_df['Eye'] == 'left']
right_eye = eye_df[eye_df['Eye'] == 'right']

# Calculate inter-pupillary distance (IPD) over time
left_pos = left_eye[['EyePosX', 'EyePosY', 'EyePosZ']].values
right_pos = right_eye[['EyePosX', 'EyePosY', 'EyePosZ']].values
ipd = np.linalg.norm(left_pos - right_pos, axis=1)
print(f"Average IPD: {np.mean(ipd)*1000:.1f} mm")

# 3. Load Projection Matrix Data
proj_df = pd.read_csv('xr_projection_matrices_2024-01-15T10-30-45.csv')

# Extract 4x4 projection matrices
def extract_projection_matrix(row):
    matrix_cols = [f'M{i}{j}' for i in range(4) for j in range(4)]
    return row[matrix_cols].values.reshape(4, 4)

# Get left eye projection matrix for first frame
left_proj_data = proj_df[proj_df['Eye'] == 'left'].iloc[0]
left_proj_matrix = extract_projection_matrix(left_proj_data)

# Extract field of view from projection matrix
fov_y = 2 * np.arctan(1.0 / left_proj_matrix[1, 1]) * 180 / np.pi
aspect_ratio = left_proj_matrix[1, 1] / left_proj_matrix[0, 0]
fov_x = fov_y * aspect_ratio
print(f"Field of View: {fov_x:.1f}° x {fov_y:.1f}°")

# 4. Load and Process XR Path Log Data
path_df = pd.read_csv('xr_path_log_2024-01-15_10-30-45_EST.csv', comment='#')

# Extract session metadata from the comment line
with open('xr_path_log_2024-01-15_10-30-45_EST.csv', 'r') as f:
    metadata_line = f.readline().strip()
    if metadata_line.startswith('# Session Metadata:'):
        metadata_parts = metadata_line.split(': ')[1].split(',')
        session_metadata = {
            'session_id': metadata_parts[0],
            'reference_space_type': metadata_parts[1],
            'units': metadata_parts[2],
            'handedness': metadata_parts[3],
            'up_axis': metadata_parts[4],
            'forward_axis': metadata_parts[5],
            'headset_height_m': float(metadata_parts[6]) if metadata_parts[6] else None
        }
        print("Session Metadata:", session_metadata)

# Filter for pose data only (ignore any other event types)
pose_data = path_df[path_df['type'] == 'pose'].copy()
pose_data['timestamp'] = pd.to_datetime(pose_data['UnixTime(ms)'], unit='ms')

# Extract position and orientation
position = pose_data[['PosX', 'PosY', 'PosZ']].values
quaternion = pose_data[['OrientW', 'OrientX', 'OrientY', 'OrientZ']].values
tracking_state = pose_data['trackingState'].values

# Filter out frames with poor tracking quality
good_tracking = pose_data['trackingState'].isin(['normal', 'limited'])
clean_position = position[good_tracking]
clean_quaternion = quaternion[good_tracking]

print(f"Total frames: {len(pose_data)}")
print(f"Good tracking frames: {len(clean_position)} ({len(clean_position)/len(pose_data)*100:.1f}%)")

# Project to floor plane (x, y coordinates)
if session_metadata['reference_space_type'] == 'local-floor':
    # Already floor-relative, just use X and Z coordinates
    floor_x = clean_position[:, 0]  # X position
    floor_y = clean_position[:, 2]  # Z position (forward/backward)
elif session_metadata['headset_height_m']:
    # Project from local space to floor
    headset_height = session_metadata['headset_height_m']
    floor_x = clean_position[:, 0]  # X position
    floor_y = clean_position[:, 2]  # Z position
    # Optionally adjust for height: floor_height = clean_position[:, 1] - headset_height
else:
    # Use raw coordinates if no height info available
    floor_x = clean_position[:, 0]
    floor_y = clean_position[:, 2]

# Extract yaw from quaternions
from scipy.spatial.transform import Rotation
head_rotation = Rotation.from_quat(clean_quaternion[:, [1,2,3,0]])  # scipy uses xyzw order
euler_angles = head_rotation.as_euler('xyz', degrees=True)
yaw = euler_angles[:, 2]  # Z-axis rotation (yaw)

# Analyze trajectory
trajectory_length = np.sum(np.sqrt(np.diff(floor_x)**2 + np.diff(floor_y)**2))
print(f"Total trajectory length: {trajectory_length:.2f} meters")

# Detect stationary periods (low movement)
velocity = np.sqrt(np.diff(floor_x)**2 + np.diff(floor_y)**2)
stationary_threshold = 0.01  # 1cm per frame
stationary_frames = velocity < stationary_threshold
print(f"Stationary frames: {np.sum(stationary_frames)} ({np.sum(stationary_frames)/len(velocity)*100:.1f}%)")

# 5. Load Derived IMU Data
imu_df = pd.read_csv('imu_derived_motion_2024-01-15T10-30-45.csv')

# Calculate sampling rate
sample_rate = len(pose_df) / (pose_df['UnixTime(ms)'].iloc[-1] - pose_df['UnixTime(ms)'].iloc[0]) * 1000
print(f"Actual sampling rate: {sample_rate:.1f} Hz")

# Extract acceleration and gyroscope data
acceleration = imu_df[['Ax', 'Ay', 'Az']].values
gyroscope = imu_df[['Gx', 'Gy', 'Gz']].values
```

## File Naming Convention

Files are automatically named with EST timestamps:
- Format: `{type}_{YYYY-MM-DD_HH-mm-ss_EST}.{ext}`
- Example: `imu_derived_motion_2024-01-15_10-30-45_EST.csv`
- Example: `raw_audio_stereo_2024-01-15_10-30-45_EST.wav`
- Timezone: Eastern Standard Time (EST) for consistent temporal reference

## Automatic Upload

Files are automatically uploaded to this repository when:
1. A valid GitHub Personal Access Token is configured
2. The "Export Data" button is clicked in the WebXR application
3. Both local download and GitHub upload are attempted

The application uploads files to the `data/` directory in the repository.
