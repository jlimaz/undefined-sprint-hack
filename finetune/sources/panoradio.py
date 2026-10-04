"""Turn the Panoradio HF dataset (S. Scholl, 2019) into training observations.

Panoradio HF is 172,800 baseband IQ vectors of 18 shortwave modes, 2,048 samples
each at 6 kHz. Bandwidth, duty cycle and frequency offset are measured from each
vector; SNR comes from the dataset's tags. The vectors carry no centre frequency,
so each one is placed on a frequency where its mode is really operated.
"""

import argparse
import collections
import csv
import os
import random
import statistics

import numpy as np

from finetune.sources.common import RAW, TRAIN, UNKNOWN_MODULATION_RATE, write_jsonl

DATASET = RAW / "panoradio" / "dataset.npy"
TAGS = RAW / "panoradio" / "tags.csv"
OUTPUT = TRAIN / "panoradio_hf.jsonl"
SEED = 20
SAMPLE_RATE_HZ = 6000
PER_MODE_AND_SNR = 80
# Below this the signal is too deep in the noise for a bandwidth measurement.
MIN_SNR_DB = 0
SMOOTH_BINS = 5
FLOOR_MARGIN = 4.0
POWER_CONTAINED = 0.95

kHz, MHz = 1e3, 1e6
_HAM_DIGITAL = [
    (3.570 * MHz, 3.600 * MHz), (7.035 * MHz, 7.045 * MHz), (7.070 * MHz, 7.125 * MHz),
    (10.130 * MHz, 10.150 * MHz), (14.070 * MHz, 14.112 * MHz), (18.095 * MHz, 18.109 * MHz),
    (21.070 * MHz, 21.110 * MHz), (28.070 * MHz, 28.150 * MHz),
]
_HAM_CW = [
    (1.810 * MHz, 1.838 * MHz), (3.500 * MHz, 3.570 * MHz), (7.000 * MHz, 7.040 * MHz),
    (10.100 * MHz, 10.130 * MHz), (14.000 * MHz, 14.070 * MHz), (21.000 * MHz, 21.070 * MHz),
    (28.000 * MHz, 28.070 * MHz),
]
_HAM_PSK = [f * MHz for f in (1.838, 3.580, 7.040, 7.070, 10.142, 14.070, 18.100, 21.070, 24.920, 28.120)]
_NAVTEX = [490 * kHz, 518 * kHz, 4209.5 * kHz]
_FAX = [f * kHz for f in (
    2618.5, 4610, 8040, 11086.5, 3855, 7880, 13882.5, 4235, 6340.5, 9110, 12750,
    4346, 8682, 12786, 17151.2, 22527, 2054, 4298, 8459, 12412.5, 9982.5, 11090, 16135,
    3622.5, 7795, 13988.5,
)]

# mode tag: (signal name, modulation, family, where it is operated, nominal bandwidth in Hz)
# Frequencies are single channels or (low, high) band segments.
# "am", "rtty50_170" and "rtty100_850" have no clear family among ours and are left out.
# "usb" and "lsb" are left out too: a 340 ms slice of speech measures about 900 Hz,
# a third of the 2.7 kHz channel that catalogues and the test file report for voice.
MODES = {
    "morse": ("Morse Code (CW)", "OOK", "amateur_radio", _HAM_CW, 100),
    "psk31": ("PSK31", "PSK", "amateur_radio", _HAM_PSK, 31),
    "psk63": ("PSK63", "PSK", "amateur_radio", _HAM_PSK, 63),
    "qpsk31": ("QPSK31", "QPSK", "amateur_radio", _HAM_PSK, 31),
    "rtty45_170": ("RTTY 45 baud 170 Hz", "FSK", "amateur_radio", _HAM_DIGITAL, 250),
    "olivia8_250": ("Olivia 8/250", "MFSK", "amateur_radio", _HAM_DIGITAL, 250),
    "olivia16_500": ("Olivia 16/500", "MFSK", "amateur_radio", _HAM_DIGITAL, 500),
    "olivia16_1000": ("Olivia 16/1000", "MFSK", "amateur_radio", _HAM_DIGITAL, 1000),
    "olivia32_1000": ("Olivia 32/1000", "MFSK", "amateur_radio", _HAM_DIGITAL, 1000),
    "dominoex11": ("DominoEX 11", "MFSK", "amateur_radio", _HAM_DIGITAL, 194),
    "mt63_1000": ("MT63-1000", "PSK", "amateur_radio", _HAM_DIGITAL, 1000),
    "navtex": ("NAVTEX (SITOR-B)", "FSK", "marine", _NAVTEX, 300),
    "fax": ("HF weather fax", "FM", "marine", _FAX, 1100),
}


def open_dataset() -> np.ndarray:
    """Memory-map the array; a partly downloaded file exposes the rows it has."""
    if not DATASET.exists():
        raise SystemExit(f"{DATASET} is missing. Run: python -m finetune.sources.fetch")
    with open(DATASET, "rb") as source:
        np.lib.format.read_magic(source)
        shape, _, dtype = np.lib.format.read_array_header_1_0(source)
        offset = source.tell()
    rows = min(shape[0], (os.path.getsize(DATASET) - offset) // (shape[1] * dtype.itemsize))
    return np.memmap(DATASET, dtype=dtype, mode="r", offset=offset, shape=(rows, shape[1]))


def measure(iq: np.ndarray) -> tuple[float, float, float]:
    """Occupied bandwidth (Hz), centre offset (Hz) and duty cycle of one vector."""
    window = np.hanning(len(iq))
    spectrum = np.fft.fftshift(np.abs(np.fft.fft(iq * window)) ** 2)
    spectrum = np.convolve(spectrum, np.ones(SMOOTH_BINS) / SMOOTH_BINS, mode="same")
    floor = np.median(spectrum)
    excess = np.clip(spectrum - FLOOR_MARGIN * floor, 0, None)
    bin_hz = SAMPLE_RATE_HZ / len(iq)
    if excess.sum() == 0:
        bandwidth, offset = 0.0, 0.0
    else:
        share = np.cumsum(excess) / excess.sum()
        tail = (1 - POWER_CONTAINED) / 2
        low, high = np.searchsorted(share, tail), np.searchsorted(share, 1 - tail)
        bandwidth = (high - low + 1) * bin_hz
        offset = ((low + high) / 2 - len(iq) / 2) * bin_hz
    envelope = np.convolve(np.abs(iq) ** 2, np.ones(32) / 32, mode="same")
    duty_cycle = float(np.mean(envelope > 0.25 * np.percentile(envelope, 95)))
    return float(bandwidth), float(offset), duty_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--per-mode-and-snr", type=int, default=PER_MODE_AND_SNR)
    args = parser.parse_args()

    data = open_dataset()
    cells = collections.defaultdict(list)
    with open(TAGS) as source:
        reader = csv.reader(source)
        next(reader)
        for index, mode, snr in reader:
            mode, snr = mode.strip(), int(snr)
            if mode in MODES and snr >= MIN_SNR_DB and int(index) < len(data):
                cells[mode, snr].append(int(index))

    rng = random.Random(SEED)
    rows = []
    measured = collections.defaultdict(list)
    for (mode, snr), indices in sorted(cells.items()):
        signal, modulation, family, channels, _ = MODES[mode]
        for index in sorted(rng.sample(indices, min(args.per_mode_and_snr, len(indices)))):
            bandwidth, offset, duty_cycle = measure(np.asarray(data[index]))
            if bandwidth == 0:
                continue
            channel = rng.choice(channels)
            frequency = rng.uniform(*channel) if isinstance(channel, tuple) else channel
            measured[mode].append(bandwidth)
            rows.append(
                {
                    "center_frequency_hz": round(frequency + offset),
                    "bandwidth_hz": max(1, round(bandwidth)),
                    "modulation": "unknown" if rng.random() < UNKNOWN_MODULATION_RATE else modulation,
                    "snr_db": float(snr),
                    "duty_cycle": round(duty_cycle, 2),
                    "label_family": family,
                    "label_signal": signal,
                    "source": "panoradio_hf",
                    "simulated": True,
                    "frequency_assigned": True,
                }
            )

    print(f"{'mode':<15} {'rows':>5} {'median Hz':>10} {'nominal Hz':>11}")
    for mode in MODES:
        if measured[mode]:
            median = statistics.median(measured[mode])
            print(f"{mode:<15} {len(measured[mode]):>5} {median:>10.0f} {MODES[mode][4]:>11}")
        else:
            print(f"{mode:<15} {0:>5} {'not downloaded yet':>22}")
    write_jsonl(OUTPUT, rows)
    print(f"wrote {len(rows)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
