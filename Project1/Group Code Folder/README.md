# Group Project Code #

Group project Code

### Steps to download data ###

go to src/init/DataExtraction.ipynb to download data.
Then select a model from /src to run the data.

### Pre-processing ###

* Sample Rate = 10000.
* Audio was trimmed, and duration of audio was 5 seconds.
* Removed all audio that was less than 5 seconds.
* Use dataset of 1000 samples.
* Balance data between male and female
* Collect and assign information of sex of speakers from SPEAKER.txt
* Computer Mel-Spectrogram (amplitude -> spectrogram -> mel-spectrogram)

# Training Model #

Models were split amongst authors

### Models used ###

* LSTM
* ResNet
* VGG
* RF