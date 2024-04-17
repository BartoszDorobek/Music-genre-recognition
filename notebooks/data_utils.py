from pathlib import Path

import librosa
import librosa.display

SAMPLE_RATE = 22_050
MAX_SIGNAL_LENGTH_TO_CROP = 660_000

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data/gtzan"
AUDIO_SAMPLES_PATH = DATA_PATH / "genres_original"
IMAGES_SAMPLES_PATH = DATA_PATH / "images_original"
AUDIO_FILE_PATH_LIST = list(AUDIO_SAMPLES_PATH.glob('**/*.wav'))
TRAIN_DATA_PATH = DATA_PATH / "genres_3_seconds"
EVALUATE_FILE_PATH = PROJECT_DIR / 'notebooks/results/test_results.csv'


def prepare_audio_stats(filename):
    y, _ = librosa.load(filename, sr=SAMPLE_RATE)
    # Trim leading and trailing silence from an audio signal
    y, _ = librosa.effects.trim(y)
    row = {
        'filename': filename.name,
        'label': filename.parent.name,
        'length': len(y)
    }
    # Total zero_crossings in our 1 song
    zero_crossings = librosa.zero_crossings(y, pad=False)
    row['zero_crossing_rate_var'] = zero_crossings.var()
    row['zero_crossing_rate_mean'] = zero_crossings.mean()

    y_harm, y_perc = librosa.effects.hpss(y)
    row["harmony_mean"] = y_harm.mean()
    row["harmony_var"] = y_harm.var()
    row["perceptr_mean"] = y_perc.mean()
    row["perceptr_var"] = y_perc.var()

    tempo, _ = librosa.beat.beat_track(y, sr=SAMPLE_RATE)
    row["tempo"] = tempo

    spectral_centroids = librosa.feature.spectral_centroid(y, sr=SAMPLE_RATE)[0]
    row["spectral_centroid_mean"] = spectral_centroids.mean()
    row["spectral_centroid_var"] = spectral_centroids.var()

    spectral_rolloff = librosa.feature.spectral_rolloff(y, sr=SAMPLE_RATE)[0]
    row["rolloff_mean"] = spectral_rolloff.mean()
    row["rolloff_var"] = spectral_rolloff.var()

    mfccs = librosa.feature.mfcc(y, sr=SAMPLE_RATE)
    for i in range(mfccs.shape[0]):
        row[f"mfcc{i + 1}_mean"] = mfccs[i, :].mean()
        row[f"mfcc{i + 1}_var"] = mfccs[i, :].var()

    chromagram = librosa.feature.chroma_stft(y, sr=SAMPLE_RATE, hop_length=5000)
    row["chroma_stft_mean"] = chromagram.mean()
    row["chroma_stft_var"] = chromagram.var()

    spec_bw = librosa.feature.spectral_bandwidth(y, sr=SAMPLE_RATE)
    row["spectral_bandwidth_mean"] = spec_bw.mean()
    row["spectral_bandwidth_var"] = spec_bw.var()

    rms = librosa.feature.rms(y)
    row["rms_mean"] = rms.mean()
    row["rms_var"] = rms.var()

    return row
