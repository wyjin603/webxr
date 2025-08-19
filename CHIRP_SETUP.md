# Chirp Audio Setup Instructions

## Adding Your Chirp File

To use your own `chirp_16_20khz_40ms.wav` file:

1. **Place the file** in the root directory of this project (same folder as `index.html`)
2. **Name it exactly**: `chirp_16_20khz_40ms.wav`
3. **File requirements**:
   - Format: WAV audio file
   - Frequency range: 16kHz to 20kHz chirp
   - Duration: 40ms (recommended)
   - Sample rate: 48kHz (recommended)

## Fallback System

If the chirp file is not found, the app will automatically:
- Generate a synthetic 16-20kHz chirp
- Duration: 40ms
- Linear frequency sweep
- Bell-shaped envelope for smooth playback

## How It Works

### Chirp Playback
- **Desktop**: Click the "🔊 Play Chirp" button
- **VR Controllers**: Press X/A button on either controller
- **Keyboard**: Press 'P' key (works in VR too)

### Recording Integration
- The chirp playback is **fully compatible** with the recording system
- When you play a chirp while recording is active, you'll capture:
  - The original chirp through speakers
  - Room acoustics and reflections
  - Any acoustic response from the environment
- Chirp events are logged with precise timestamps for analysis

### Acoustic Analysis Features
- High-precision timestamp logging for chirp events
- Compatible with 48kHz audio recording
- Chirp events stored in metadata for post-processing
- Perfect for:
  - Room impulse response measurements
  - Acoustic delay analysis
  - Multi-modal sensor fusion studies
  - Echo and reverberation analysis

## Example Use Case

1. Start audio recording
2. Start WebXR session for head tracking
3. Play chirp at specific moments
4. Move around in VR space
5. Export all data (XR tracking + audio + chirp timestamps)
6. Analyze acoustic properties vs. head position

This creates a comprehensive dataset linking spatial movement with acoustic measurements!
