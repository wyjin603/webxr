/**
 * AudioWorklet Processor for Raw PCM Audio Capture
 * Captures raw audio samples at 48kHz for high-quality WAV export
 *
 * Timing contract
 * ---------------
 * The exported WAV is indexed in samples, and every consumer (chirp arrival
 * times, pose registration) assumes sample index N means N/sampleRate seconds
 * after capture started. That only holds if the stream never loses frames.
 *
 * `currentFrame` advances with the AudioContext's own sample clock whether or
 * not this callback ran, so comparing it against the number of frames actually
 * stored detects missed render quanta exactly. Missing frames are filled with
 * silence, keeping sample index proportional to elapsed time, and every gap is
 * reported to the main thread so it lands in the timing log.
 *
 * Earlier versions did neither: they pushed samples one at a time into growing
 * JS arrays and then spliced 100k elements off the front on the audio thread.
 * That overruns the 2.67 ms render deadline, which drops render quanta, and
 * with no frame accounting the dropped audio silently shortened the file. A
 * staircase of such losses read out as a ~2800 ppm sample-clock drift that was
 * never there. Keep this callback allocation-light and O(1) per frame.
 */
const CHUNK_FRAMES = 16384;          // ~0.34 s at 48 kHz; transferred, not copied
const MAX_FILL_FRAMES = 48000 * 5;   // refuse to invent more than 5 s of silence

class AudioCaptureProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.chunkFrames = CHUNK_FRAMES;
        this.left = new Float32Array(this.chunkFrames);
        this.right = new Float32Array(this.chunkFrames);
        this.fill = 0;
        this.isRecording = false;
        this.sampleRate = 48000; // Will be updated with actual sample rate
        this.channelCount = 1;   // Will be updated
        this.startFrame = null;
        this.framesWritten = 0;
        this.framesDropped = 0;
        this.dropEvents = 0;
        this.overflow = false;

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
        this.fill = 0;
        this.startFrame = null;
        this.framesWritten = 0;
        this.framesDropped = 0;
        this.dropEvents = 0;
        this.overflow = false;

        // Send confirmation back to main thread
        this.port.postMessage({
            type: 'recordingStarted',
            sampleRate: this.sampleRate,
            channelCount: this.channelCount,
            contextSampleRate: sampleRate,
            ctxTime: currentTime,
            currentFrame: currentFrame
        });
    }

    stopRecording() {
        if (!this.isRecording) return;
        this.isRecording = false;
        this.flush(true);
        this.port.postMessage({
            type: 'captureSummary',
            expectedFrames: this.startFrame === null ? 0 : (currentFrame - this.startFrame),
            storedFrames: this.framesWritten,
            filledFrames: this.framesDropped,
            dropEvents: this.dropEvents,
            overflow: this.overflow,
            ctxTime: currentTime,
            currentFrame: currentFrame,
            contextSampleRate: sampleRate
        });
    }

    /** Hand the current chunk to the main thread, transferring ownership. */
    flush(isFinal) {
        const n = this.fill;
        if (n === 0 && !isFinal) return;
        const l = (n === this.left.length) ? this.left : this.left.slice(0, n);
        const r = (n === this.right.length) ? this.right : this.right.slice(0, n);
        this.port.postMessage({
            type: isFinal ? 'recordingStopped' : 'audioChunk',
            samplesLeft: l,
            samplesRight: r,
            sampleRate: this.sampleRate,
            channelCount: this.channelCount,
            duration: this.framesWritten / (this.sampleRate || 48000)
        }, [l.buffer, r.buffer]);
        // ownership was transferred away, so take fresh backing stores
        this.left = new Float32Array(this.chunkFrames);
        this.right = new Float32Array(this.chunkFrames);
        this.fill = 0;
    }

    /** Append one render quantum; flushes whenever a chunk fills up. */
    writeBlock(lsrc, rsrc, n) {
        let off = 0;
        while (off < n) {
            const room = this.chunkFrames - this.fill;
            const take = Math.min(room, n - off);
            if (lsrc) {
                this.left.set(lsrc.subarray(off, off + take), this.fill);
                this.right.set((rsrc || lsrc).subarray(off, off + take), this.fill);
            } else {
                this.left.fill(0, this.fill, this.fill + take);
                this.right.fill(0, this.fill, this.fill + take);
            }
            this.fill += take;
            off += take;
            if (this.fill === this.chunkFrames) this.flush(false);
        }
        this.framesWritten += n;
    }

    process(inputs, outputs, parameters) {
        if (!this.isRecording) {
            return true;
        }

        if (this.startFrame === null) {
            this.startFrame = currentFrame;
        }

        // Re-align to the context clock before storing anything. A shortfall
        // here means render quanta were missed and that audio no longer exists;
        // silence keeps later samples at their true positions in the file.
        const expected = currentFrame - this.startFrame;
        if (expected > this.framesWritten) {
            let missing = expected - this.framesWritten;
            if (missing > MAX_FILL_FRAMES) {
                this.overflow = true;
                missing = MAX_FILL_FRAMES;
            }
            this.dropEvents++;
            this.framesDropped += missing;
            this.port.postMessage({
                type: 'captureGap',
                atFrame: this.framesWritten,
                missing: missing,
                ctxTime: currentTime,
                currentFrame: currentFrame
            });
            this.writeBlock(null, null, missing);
        }

        const input = inputs[0];
        if (!input || input.length === 0) {
            // No input this quantum. Do not guess its length: the gap check on
            // the next call will fill exactly what the clock says was missed.
            return true;
        }
        const leftChannel = input[0];
        if (!leftChannel || leftChannel.length === 0) {
            return true;
        }
        const rightChannel = (input.length > 1 && input[1] && input[1].length === leftChannel.length)
            ? input[1]
            : null; // mono input: duplicate left into right
        this.writeBlock(leftChannel, rightChannel, leftChannel.length);

        return true; // Keep processor alive
    }
}

// Register the processor
registerProcessor('audio-capture-processor', AudioCaptureProcessor);
