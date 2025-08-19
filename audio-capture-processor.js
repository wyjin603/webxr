/**
 * AudioWorklet Processor for Raw PCM Audio Capture
 * Captures raw audio samples at 48kHz for high-quality WAV export
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 4096; // Chunk size for efficient processing
        this.sampleBuffer = [];
        this.isRecording = false;
        this.sampleRate = 48000; // Will be updated with actual sample rate
        
        // Listen for messages from main thread
        this.port.onmessage = (event) => {
            const { command, data } = event.data;
            
            switch (command) {
                case 'start':
                    this.startRecording(data);
                    break;
                case 'stop':
                    this.stopRecording();
                    break;
                case 'setSampleRate':
                    this.sampleRate = data.sampleRate;
                    break;
            }
        };
    }
    
    startRecording(config = {}) {
        this.isRecording = true;
        this.sampleBuffer = [];
        console.log('[AudioWorklet] Started raw PCM recording at', this.sampleRate, 'Hz');
        
        // Send confirmation back to main thread
        this.port.postMessage({
            type: 'recordingStarted',
            sampleRate: this.sampleRate
        });
    }
    
    stopRecording() {
        this.isRecording = false;
        console.log('[AudioWorklet] Stopped recording, captured', this.sampleBuffer.length, 'samples');
        
        // Send all captured samples back to main thread
        this.port.postMessage({
            type: 'recordingStopped',
            samples: new Float32Array(this.sampleBuffer),
            sampleRate: this.sampleRate,
            duration: this.sampleBuffer.length / this.sampleRate
        });
        
        // Clear buffer to free memory
        this.sampleBuffer = [];
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        // Only process if we have input and are recording
        if (!this.isRecording || !input || input.length === 0) {
            return true;
        }
        
        // Get the first channel (mono recording)
        const channelData = input[0];
        
        if (channelData && channelData.length > 0) {
            // Copy samples to our buffer
            for (let i = 0; i < channelData.length; i++) {
                this.sampleBuffer.push(channelData[i]);
            }
            
            // Periodically send chunks to main thread to avoid memory issues
            if (this.sampleBuffer.length >= this.bufferSize * 50) { // Send every ~200KB of samples
                const chunkSize = this.bufferSize * 25; // Send smaller chunks more frequently
                this.port.postMessage({
                    type: 'audioChunk',
                    samples: new Float32Array(this.sampleBuffer.splice(0, chunkSize)),
                    sampleRate: this.sampleRate
                });
            }
        }
        
        return true; // Keep processor alive
    }
}

// Register the processor
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
