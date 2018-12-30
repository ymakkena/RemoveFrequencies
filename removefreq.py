#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from scipy.fftpack import fft, fftfreq
from sound import Sound, DepthException
import re

parser = argparse.ArgumentParser(prog='removefreq.py', description='Automatically detect and filter persistent high frequencies')
parser.add_argument("--break-on-failure", dest="break_on_failure", help='If a file cannot be read, terminate', action="store_true")
parser.add_argument("files", metavar='filename', type=str, nargs='+', help='Files to read')

args = parser.parse_args()

plt.ion() # Make the plot interactive
# Allows us to update it in the code

FREQUENCY_MAXIMUM_DIFFERENTIATING_DIFFERENCE = 500 # Hz
FREQUENCY_RFREQ_MARGIN = 50 # Hz
FREQUENCY_MINIMUM_COUNT = 50
FREQUENCY_MINIMUM = 12000 # Hz
# Sounds under this frequency will not be removed
FREQUENCY_HARD_MINIMUM_RATIO = 2
# Frequencies less than the nyqsuit frequency divided
# by this ratio are not considered in calculations
AUDIO_DAMPING_SCALE = 1.01 # Prevent peaks

FFT_SAMPLE_SIZE = 2000 # FFT sample size in milliseconds

DEBUG = False
SHOW_FFT = False
BREAK_ON_SKIPS = False


def gen_output_filename(filename):
    pattern = re.compile("\.[^.]+$")
    return pattern.sub("-out.wav", filename)


def process(filename):
    try:
        fs, snd = wavfile.read(filename)
        sndobj = Sound(fs, snd)

    except (ValueError, DepthException) as e:
        print("{}".format(e))
        print("File '{}' could not be read".format(filename))
        print("Convert this file to either 16-bit or 32-bit PCM WAV file format")
        if BREAK_ON_SKIPS: return False
        print("Skipping...")
        return True

    print("==== Information about {} ====".format(filename))
    print("Samples: {}".format(sndobj.samples))
    print("Channels: {}".format(sndobj.channels))
    print("Duration: {}".format(sndobj.duration))
    print("Depth: {}-bit".format(sndobj.depth))
    print("============================"+"="*len(filename))

    signals_to_save = np.array([], dtype=snd.dtype).reshape(sndobj.samples, 0)
    for c in range(sndobj.channels):
        print("Parsing channel {}".format(c))
        cleaned_signal = parse(snd.T[c], sndobj)
        signals_to_save = np.column_stack([signals_to_save, cleaned_signal])

    if DEBUG:
        print("-------------")
        print(signals_to_save)
        print(len(signals_to_save))
        print(sndobj.channels)
        print(len(cleaned_signal))
        print(len(signals_to_save[0]))
        print("-------------")

    output_filename = gen_output_filename(filename)
    print("Saving to {}".format(output_filename))
    wavfile.write(output_filename, rate=sndobj.fs, data=signals_to_save)

    return True


def find_outstanding_frequencies(data, sndobj, points_per_sample):
    # THRESHOLD = 5000000
    diff_data = np.diff(data)
    low_band = int(points_per_sample/(2 * FREQUENCY_HARD_MINIMUM_RATIO))
    # Above this value are frequencies that are considered for removal

    num_outstanding = 3

    # Find the frequencies of highest difference
    low_ind = np.argpartition(diff_data[low_band:], -num_outstanding)[-num_outstanding:] + low_band
    # Find the frequencies of most negative difference
    high_ind = np.argpartition(-diff_data[low_band:], -num_outstanding)[-num_outstanding:] + low_band

    # Normalize
    low_ind = (low_ind * sndobj.fs)//points_per_sample
    high_ind = (high_ind * sndobj.fs)//points_per_sample

    # np.where(data[high_ind] > THRESHOLD * points_per_sample, high_ind, 0)

    if DEBUG:
        sys.stdout.write("h indices: ")
        print(high_ind)
        sys.stdout.write("l indeces: ")
        print(low_ind)
        # print(fs/points_per_sample * data[high_ind])

        # Convert back to graph units
        if SHOW_FFT:
            for x in (high_ind * points_per_sample)//sndobj.fs:
                plt.axvline(x=x, color='b')
            for x in (low_ind * points_per_sample)//sndobj.fs:
                plt.axvline(x=x, color='g')

    ret = []
    for ind1 in high_ind:
        for ind2 in low_ind:
            if(ind1 < FREQUENCY_MINIMUM or ind2 < FREQUENCY_MINIMUM): continue
            if abs(ind1 - ind2) < FREQUENCY_MAXIMUM_DIFFERENTIATING_DIFFERENCE:
                ret.append((ind1, ind2) if ind2 > ind1 else(ind2, ind1))

    return ret


def extract_removefreq_frequencies(cand):
    candidates = np.array(cand)
    candidates[:,1] += FREQUENCY_RFREQ_MARGIN
    candidates[:,0] -= FREQUENCY_RFREQ_MARGIN

    final = []
    for c in np.array(candidates):
        avg = (c[0] + c[1])/2
        placed = False
        for f in final:
            favg = (f["data"][0] + f["data"][1])/(2 * f["count"])
            if abs(avg - favg) < FREQUENCY_MAXIMUM_DIFFERENTIATING_DIFFERENCE:
                f["count"] += 1
                f["data"] += c
                placed = True
                break
        if placed: continue
        final.append({
            "data": c,
            "count": 1
        });

    ret = []
    for f in final:
        if f["count"] > FREQUENCY_MINIMUM_COUNT:
            ret.append(f["data"]/f["count"])

    return ret


def removefreq(frequencies, sound_data, sndobj, order=1):
    nyq = sndobj.fs * 0.5
    for ft in frequencies:
        num, denom = butter(order,[ft[0]/nyq,ft[1]/nyq], btype='removefreq', analog=False) # Bandstop filters
        cleaned_data = lfilter(num, denom, sound_data).astype(sndobj.snd.dtype)
        sound_data = cleaned_data
    return sound_data # Cleaned


def parse(integer_data, sndobj):
    sound_data = integer_data/AUDIO_DAMPING_SCALE
    points_per_sample = FFT_SAMPLE_SIZE*sndobj.fs//1000
    # frequency_data = fftfreq(points_per_sample, FFT_SAMPLE_SIZE/1000)*points_per_sample * FFT_SAMPLE_SIZE/1000
    # frequency_conversion_ratio = fs/points_per_sample
    candidate_frequencies = []
    for i in range(0, sndobj.samples, points_per_sample):
        # FFT data
        fft_data = fft(sound_data[i:(i+points_per_sample)])
        real_length = len(fft_data)//2
        # We only care about the first half; the other half is the same
        real_data = abs(fft_data[:(real_length-1)])

        if SHOW_FFT:
            # Plot the data as a frequency spectrum
            plt.plot(real_data,'r')
            plt.show()
            plt.pause(0.2)
            plt.clf()

        sample_freqs = find_outstanding_frequencies(real_data, sndobj, points_per_sample)
        # Tuple of high and low bands
        candidate_frequencies += sample_freqs

    removefreq_frequencies = extract_removefreq_frequencies(candidate_frequencies)
    if not removefreq_frequencies:
        print("No frequencies to remove...")
        return sound_data

    print("Removing the following frequencies:")
    for ft in removefreq_frequencies:
        print("{}-{}Hz".format(*ft))

    cleaned = removefreq(removefreq_frequencies, sound_data, sndobj)
    return cleaned


if args.break_on_failure:
    BREAK_ON_SKIPS = True

print(args.files)
for f in args.files:
    if not process(f):
        print("Terminating...")
