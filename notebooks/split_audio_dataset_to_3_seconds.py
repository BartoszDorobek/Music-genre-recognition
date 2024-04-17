import glob
from pathlib import Path

from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm

ORG_AUDIO_PATH = "../data/gtzan/genres_original/*/*.wav"

file_path_list = glob.glob(ORG_AUDIO_PATH, recursive=True)

chunk_length_ms = 3000  # pydub calculates in millisec


def process_audio(file_path):
    myaudio = AudioSegment.from_file(file_path, "wav")
    chunks = make_chunks(myaudio, chunk_length_ms)  # Make chunks of one sec
    for i, chunk in enumerate(chunks):
        if chunk.duration_seconds >= 3:
            chunk_path = file_path.replace('genres_original', 'genres_3_seconds').replace('.wav', "_{0}.wav".format(i))
            Path(chunk_path).parent.mkdir(parents=True, exist_ok=True)
            chunk.export(chunk_path, format="wav")


if __name__ == "__main__":
    for file_path in tqdm(file_path_list):
        process_audio(file_path)
