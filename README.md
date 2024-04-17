# Music genre recognition with AI

# 1. Machine learning models

Datasource was a tabular representation with extracted audio features:

![raw_data_2](https://github.com/BartoszDorobek/Music-genre-recognition/assets/53353490/0dc1c8e5-05f5-4359-a1d6-14a1d0ec1edb)

I chose a Bayesian search method to find the best hyperparameters for each algorithm.

![modele_acc](https://github.com/BartoszDorobek/Music-genre-recognition/assets/53353490/92ec36bf-1276-4d05-ae8c-2283f812fb7b)

Results were compared for multiple ML algorithms and for two different audio sample lengths.

![image](https://github.com/BartoszDorobek/Music-genre-recognition/assets/53353490/4aa54add-4dc8-4580-a5dd-0a408b6b8174)

The best accuracy (91,24%) was achieved by the LightGBM model trained on 3-second audio.

![macierz_pomylek_3](https://github.com/BartoszDorobek/Music-genre-recognition/assets/53353490/600f361f-794f-4e1a-840a-cc28b31caeea)

# 2. Convolutional Neutral Network

For the second part of the project I used Mel-spectrogram representation of audio.

![image](https://github.com/BartoszDorobek/Music-genre-recognition/assets/53353490/6f07e31e-bb64-467a-8189-e8af26c7c184)

I compared many CNN architectures to choose the best one, which I present below and which achieved an accuracy level of 85,06%.

![image](https://github.com/BartoszDorobek/Music-genre-recognition/assets/53353490/bf3fd0ac-214a-4b73-8d0d-40b9ac45f189)

![train_val_acc](https://github.com/BartoszDorobek/Music-genre-recognition/assets/53353490/90d11e52-c5ca-4f18-bb20-648368d5e42e)
