# Neural Machine Translation with Attention Mechanism
This work is done as part of [assignment](https://sites.google.com/site/2019e1246/schedule/assignment-2) 
for [E1 246: Natural Language Understanding (2019)](https://sites.google.com/site/2019e1246/basics). The report for the same can be found 
[here](https://github.com/rv-chittersu/machine-translation/blob/master/report.pdf)


## File Structure
Project layout
```
data/
- proc.train.tsv
- pros.dev.tsv
- proc.test.tsv
- en.vocab
- de.vocab
results/
Readme.md
attention_handler.py
config.ini
config_handler.py
decoder.py
driver.py
encoder.py
preprocessing.py
requirements.txt
training_handler.py
utils.py
report.pdf
```

### Data

**proc.train.tsv, proc.dev.tsv and proc.test.tsv** are generated from **train.tsv, dev.tsv and test.tsv** respectively. They have sentences represented in encoded form which can be directly fed to model. 

The encodings in the above file is generated based on **en.vocab, de.vocab** which is built using **train.tsv**

*Note:* New training splits and vocabulary can be generated from *preprocessing.py* but data sets need to be procured by the user
*Note:2* **train.tsv** isn't uploaded because of size issue. One can later add it as mentioned in below sections
### Code

**config_handler.py** acts as interface between *config.ini* and the rest of the program.<br>
**utils.py** contains generic utility functions used by rest of the program<br>
**preprocessing.py** is used to generate encoded train, dev and test from parallel corpus. As intermediate steps it generates processed datasets and vocabulary. All the expected input and files should be modified in *config.ini*  <br>
**encoder.py** contains code for encoder layer<br>
**decoder.py** contains code for decoder layer<br>
**attention_handler.py** contains code for various attention types.<br>
**training_handler.py** is used to manage training/validation/testing batch wise<br>
**driver.py** is entry point for the process and manages program life cycle.<br>

### Results


The files generated in a run are stored in *ResultsDirectory* specified in configs file<br> 
The files generated on each run will have unique **key**

The following files will be generated in after training
* results file(which have concise information of the run) of form **[key].results**

## Config File
Contains all adjustable parameters settings for program to run. The whole code doesn't take any additional parameters. So for functionality always keep config file consistent.
<details>
<summary>
[Format of the Config file]
</summary>

```
[DATA]
; These files will be used as datasets
SourceData = ./data.en.de/data.en
SourceDevData = ./data.en.de/dev-data.en
SourceTestData = ./data.en.de/test-data.en
DestinationData = ./data.en.de/data.de
DestinationTestData = ./data.en.de/test-data.de
DestinationDevData = ./data.en.de/dev-data.de
MaxSentenceLength = 20

; These files are generated by preprocessor and later used by driver
TrainingData = ./data.en.de/train.tsv
DevData = ./data.en.de/dev.tsv
TestData = ./data.en.de/test.tsv

; These files are generated by preprocessor and later used by driver
ProcessedTrainingData = ./data.en.de/proc.train.tsv
ProcessedDevData = ./data.en.de/proc.dev.tsv
ProcessedTestData = ./data.en.de/proc.test.tsv

SourceLang = en
DestinationLang = de

; generated by preprocessor
SourceVocab = ./data.en.de/en.vocab
DestinationVocab = ./data.en.de/de.vocab

MinimumFrequency = 10

ResultDir = ./results
CheckpointDir = ./models

[MODEL]
LSTMHiddenUnits = 64
LSTMLayers = 2
SourceEmbeddingDim = 128
DestinationEmbeddingDim = 128
LearningRate = 0.01
MaxDecodeLength = 15

[ATTENTION]
EncoderAttention = additive
DecoderAttention = additive
KeyValueSplit = 48,16

SelfAttention = True
AttentionHeads = 2

[TRAINING]
Epochs = 5
TrainingBatchSize = 300
MaxTrainingBatches = 600
DevBatchSize = 200
MaxDevBatches = 100
TestBatchSize = 200
MaxTestBatches = 150
```

</details>

## How to Run

create a python3 virtual environment.<br>
In the virtual environment run following to install required packages

```bash
pip3 install -r requirements.txt
```

### Run on available processed dataset
verify that files in config are present and run following commands from project folder
```
python driver.py
``` 

### Generate new training split
update *DATA* section of *config.ini* with new dataset path.<br>

*param1* - training set size (optional)<br>
```
python preprocessing.py 
```

and run
```
python driver.py
```
