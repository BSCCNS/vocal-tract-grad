from gfmdriver import GFMDriver
import cProfile
import soundfile as sf
import pstats

driver = GFMDriver()
print("Reading...")
audio, samplerate = sf.read("./C2.wav")
driver.store_audio(audio, samplerate)

print("Profiling...")
cProfile.run('driver.play_audio(False)','restats')

p=pstats.Stats('restats')

