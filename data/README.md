# WebXR Data Collection Files

This directory contains data files collected from the WebXR IMU & Acoustic Data Collector application.

## File Types

### IMU Data Files (`imu_data_*.csv`)
- **Format**: CSV with Unix timestamps and sensor readings
- **Columns**: UnixTime(ms), Ax, Ay, Az, Gx, Gy, Gz, QuatW, QuatX, QuatY, QuatZ
- **Sample Rate**: Target 800Hz
- **Units**: 
  - Acceleration (Ax, Ay, Az): m/s²
  - Gyroscope (Gx, Gy, Gz): rad/s
  - Quaternion (QuatW, QuatX, QuatY, QuatZ): normalized

### Audio Files (`audio_data_*.webm`)
- **Format**: WebM audio container with Opus codec
- **Sample Rate**: 48kHz
- **Channels**: Mono (1 channel)
- **Quality**: 128kbps bitrate

### Session Metadata (`session_metadata_*.json`)
- **Format**: JSON
- **Contents**: Session duration, sample counts, device info, exported files list

## Data Processing

### Converting Audio Files
To convert WebM audio files to WAV format for analysis:

```bash
# Using FFmpeg
ffmpeg -i audio_data_2024-01-15T10-30-45.webm -ar 48000 -ac 1 audio_data_2024-01-15T10-30-45.wav

# Using Python (requires pydub)
from pydub import AudioSegment
audio = AudioSegment.from_file("audio_data_2024-01-15T10-30-45.webm")
audio.export("audio_data_2024-01-15T10-30-45.wav", format="wav")
```

### Processing IMU Data
IMU data can be processed with standard CSV libraries:

```python
import pandas as pd
import numpy as np

# Load IMU data
df = pd.read_csv('imu_data_2024-01-15T10-30-45.csv')

# Convert Unix timestamp to datetime
df['timestamp'] = pd.to_datetime(df['UnixTime(ms)'], unit='ms')

# Calculate sampling rate
sample_rate = len(df) / (df['UnixTime(ms)'].iloc[-1] - df['UnixTime(ms)'].iloc[0]) * 1000
print(f"Actual sampling rate: {sample_rate:.1f} Hz")

# Extract acceleration and gyroscope data
acceleration = df[['Ax', 'Ay', 'Az']].values
gyroscope = df[['Gx', 'Gy', 'Gz']].values
quaternion = df[['QuatW', 'QuatX', 'QuatY', 'QuatZ']].values
```

## File Naming Convention

Files are automatically named with ISO timestamps:
- Format: `{type}_data_{YYYY-MM-DDTHH-mm-ss}.{ext}`
- Example: `imu_data_2024-01-15T10-30-45.csv`

## Automatic Upload

Files are automatically uploaded to this repository when:
1. A valid GitHub Personal Access Token is configured
2. The "Export Data" button is clicked in the WebXR application
3. Both local download and GitHub upload are attempted

The application uploads files to the `data/` directory in the repository.
