/**
 * AudioWorklet Processor for Raw PCM Audio Capture
 * Captures raw audio samples at 48kHz for high-quality WAV export
 */
class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.bufferSize = 4096; // Chunk size for efficient processing
        this.sampleBufferLeft = [];
        this.sampleBufferRight = [];
        this.isRecording = false;
        this.sampleRate = 48000; // Will be updated with actual sample rate
        this.channelCount = 1; // Will be updated
        
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
                    this.channelCount = data.channelCount || 1;
                    break;
            }
        };
    }
    
    startRecording(config = {}) {
        this.isRecording = true;
        this.sampleBufferLeft = [];
        this.sampleBufferRight = [];
        console.log('[AudioWorklet] Started raw PCM recording at', this.sampleRate, 'Hz', this.channelCount, 'channels');
        
        // Send confirmation back to main thread
        this.port.postMessage({
            type: 'recordingStarted',
            sampleRate: this.sampleRate,
            channelCount: this.channelCount
        });
    }
    
    stopRecording() {
        this.isRecording = false;
        console.log('[AudioWorklet] Stopped recording, captured Left:', this.sampleBufferLeft.length, 'Right:', this.sampleBufferRight.length, 'samples');
        
        // Send all captured samples back to main thread
        this.port.postMessage({
            type: 'recordingStopped',
            samplesLeft: new Float32Array(this.sampleBufferLeft),
            samplesRight: new Float32Array(this.sampleBufferRight),
            sampleRate: this.sampleRate,
            channelCount: this.channelCount,
            duration: this.sampleBufferLeft.length / this.sampleRate
        });
        
        // Clear buffers to free memory
        this.sampleBufferLeft = [];
        this.sampleBufferRight = [];
    }
    
    process(inputs, outputs, parameters) {
        const input = inputs[0];
        
        // Only process if we have input and are recording
        if (!this.isRecording || !input || input.length === 0) {
            return true;
        }
        
        // Process all available channels (stereo recording)
        const numChannels = Math.min(input.length, 2); // Support up to 2 channels
        
        if (numChannels > 0) {
            // Left channel (always present)
            const leftChannel = input[0];
            if (leftChannel && leftChannel.length > 0) {
                for (let i = 0; i < leftChannel.length; i++) {
                    this.sampleBufferLeft.push(leftChannel[i]);
                }
            }
            
            // Right channel (if available, otherwise duplicate left)
            if (numChannels > 1 && input[1]) {
                const rightChannel = input[1];
                for (let i = 0; i < rightChannel.length; i++) {
                    this.sampleBufferRight.push(rightChannel[i]);
                }
            } else {
                // Duplicate left channel for mono input
                for (let i = 0; i < leftChannel.length; i++) {
                    this.sampleBufferRight.push(leftChannel[i]);
                }
            }
            
            // Periodically send chunks to main thread to avoid memory issues
            if (this.sampleBufferLeft.length >= this.bufferSize * 50) { // Send every ~200KB of samples
                const chunkSize = this.bufferSize * 25; // Send smaller chunks more frequently
                this.port.postMessage({
                    type: 'audioChunk',
                    samplesLeft: new Float32Array(this.sampleBufferLeft.splice(0, chunkSize)),
                    samplesRight: new Float32Array(this.sampleBufferRight.splice(0, chunkSize)),
                    sampleRate: this.sampleRate,
                    channelCount: this.channelCount
                });
            }
        }
        
        return true; // Keep processor alive
    }
}

// Register the processor
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
