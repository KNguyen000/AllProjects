Due to the confidentiality agreement with Logickube, I cannot link or show the data to this project

### Pre-processing ###
* Transform data into customer path: Facebook -> Google -> Youtube -> Organic.
* Assign each customer path a conversion (Eiher 0 or 1)
* Split the data into even conversions
* Split the data 70/30, train/test sets
* Tokenise touchpoints (ie. Facebook, Youtube etc.) into assigned numbers

### Modelling ###
* Use a bidirectional LSTM
* Train model