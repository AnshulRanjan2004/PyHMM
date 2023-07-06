import pyaudio
import wave

def record_audio(duration, output_file):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100

    audio = pyaudio.PyAudio()

    stream = audio.open(format=format,
                        channels=channels,
                        rate=rate,
                        input=True,
                        frames_per_buffer=chunk)

    print("Recording audio...")

    frames = []
    for i in range(0, int(rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wave_file = wave.open(output_file, 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(audio.get_sample_size(format))
    wave_file.setframerate(rate)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()

# Usage: record_audio(duration_in_seconds, output_file_name)

# Example usage: Record audio for 5 seconds and save it as 'recording.wav'
record_audio(5, 'recording.wav')
