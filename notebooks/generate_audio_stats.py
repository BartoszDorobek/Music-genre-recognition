import warnings
from datetime import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data_utils import prepare_audio_stats

warnings.simplefilter("ignore")

if __name__ == "__main__":
    audio_dataset_path = Path("../data/gtzan/genres_original/")
    audio_file_path_list = list(audio_dataset_path.glob('**/*.wav'))
    df = pd.DataFrame([])
    for audio_file_path in tqdm(audio_file_path_list):
        data = pd.DataFrame.from_dict([prepare_audio_stats(audio_file_path)])
        df = df.append(data)

    RESULTS_PATH = Path('../results/gtzan')
    curent_timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    df.to_csv(RESULTS_PATH / f'audio_stats_data_{curent_timestamp}.csv', index=False)
