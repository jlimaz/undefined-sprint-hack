"""Hand-curated drone control and video links, as training observations.

The signal catalogue has no drone links, and the test file's ten are not used
here. These entries are approximate figures from public product documentation,
not measurements: treat this file as a stopgap until real captures replace it.
"""

import argparse
import random

from finetune.sources.common import TRAIN, observe, test_signal_names, write_jsonl

OUTPUT = TRAIN / "drone_links.jsonl"
SEED = 20
ROWS = 1200

MHz = 1_000_000
_BAND_2G4 = [2400 * MHz, 2483 * MHz]
_BAND_5G8 = [5725 * MHz, 5850 * MHz]
_BAND_900 = [902 * MHz, 928 * MHz]
_BAND_868 = [863 * MHz, 870 * MHz]
_BAND_433 = [433 * MHz, 435 * MHz]

# name: (frequency band, bandwidths in Hz, modulations)
LINKS = {
    "FrSky ACCST 2.4 GHz control link": (_BAND_2G4, [1 * MHz], ["GFSK"]),
    "FlySky AFHDS 2A control link": (_BAND_2G4, [500_000], ["GFSK"]),
    "Spektrum DSMX control link": (_BAND_2G4, [1 * MHz], ["DSSS"]),
    "Futaba FASST control link": (_BAND_2G4, [1_500_000], ["GFSK"]),
    "ImmersionRC Ghost 2.4 GHz control link": (_BAND_2G4, [800_000], ["LoRa", "FSK"]),
    "TBS Tracer 2.4 GHz control link": (_BAND_2G4, [800_000], ["LoRa", "FSK"]),
    "mLRS 2.4 GHz control link": (_BAND_2G4, [800_000], ["LoRa"]),
    "FrSky R9 900 MHz control link": (_BAND_900, [500_000], ["LoRa"]),
    "FrSky R9 868 MHz control link": (_BAND_868, [250_000, 500_000], ["LoRa"]),
    "mLRS 900 MHz control link": (_BAND_900, [500_000], ["LoRa"]),
    "RFD900 telemetry radio": (_BAND_900, [250_000, 500_000], ["GFSK"]),
    "Dragon Link 433 MHz control link": (_BAND_433, [250_000], ["FSK", "GFSK"]),
    "DJI Lightbridge video and control 2.4 GHz": (_BAND_2G4, [10 * MHz], ["OFDM"]),
    "DJI O3 video and control 5.8 GHz": (_BAND_5G8, [10 * MHz, 20 * MHz], ["OFDM"]),
    "Autel SkyLink video and control 2.4 GHz": (_BAND_2G4, [10 * MHz, 20 * MHz], ["OFDM"]),
    "Walksnail Avatar digital video 5.8 GHz": (_BAND_5G8, [20 * MHz], ["OFDM"]),
    "Herelink video and control 2.4 GHz": (_BAND_2G4, [10 * MHz, 20 * MHz], ["OFDM"]),
    "Analog FPV video 1.3 GHz": ([1240 * MHz, 1300 * MHz], [18 * MHz], ["FM"]),
    "Analog FPV video 2.4 GHz": (_BAND_2G4, [18 * MHz], ["FM"]),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=int, default=ROWS)
    args = parser.parse_args()

    clash = set(LINKS) & test_signal_names()
    if clash:
        raise SystemExit(f"These links are in the test file and must not be trained on: {clash}")
    rng = random.Random(SEED)
    each = -(-args.rows // len(LINKS))
    rows = [
        {
            **observe(rng, frequencies, bandwidths, modulations),
            "label_family": "drone_link",
            "label_signal": name,
            "source": "curated",
            "simulated": True,
        }
        for name, (frequencies, bandwidths, modulations) in LINKS.items()
        for _ in range(each)
    ]
    write_jsonl(OUTPUT, rows)
    print(f"wrote {len(rows)} rows for {len(LINKS)} drone links to {OUTPUT}")


if __name__ == "__main__":
    main()
