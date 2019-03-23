# Neural Machine Translation with Attention Mechanism
This work is done as part of [assignment](https://sites.google.com/site/2019e1246/schedule/assignment-2) 
for [E1 246: Natural Language Understanding (2019)](https://sites.google.com/site/2019e1246/basics). The report for the same can be found [here](https://github.com/rv-chittersu/word2vec/blob/master/report.pdf)

## Data



## File Structure
Project layout
```
data/
- default.split.txt
- default.vocab
- SimLex-999.txt
- questions-words.txt
results/
config.ini
initialize.py
driver.py
model.py
data_handler.py
config_handler.py
utils.py
similar_words.py
simlex_results.py
analogical_task.py
requirements.txt
report.pdf
```

### Data

This folder holds pre-generated data which is necessary for training the model. 

**SimLex-999.txt** is used for evaluation and is downloaded from [here](https://fh295.github.io/simlex.html).<br>
**questions-words.txt** is used to evaluate analogical reasoning task.
**default.split.txt** contains file ids split for train validation and testing.<br>
**default.vocab** contains vocabulary generated on training data.<br>

*Note:* New training splits and vocabulary can be generated from *initialize.py*

### Code

**initialize.py** downloads nltk corpus. With arguments it can be used to generate training-validation split and vocabulary<br>
**driver.py** contains main function for training the model.<br>
**model.py** holds logic for word2vec skipgram model.<br>
**data_handler.py** is for generating inputs for model from corpus.<br>
**config_handler.py** is interface for reading config file.<br>
**utils.py** has implementation of functions used by other files.<br>
**similar_words.py** generates similar words for a given word based on embeddings.<br>
**simlex_results.py** calculates co-relation between simlex-999 scores and model's similarity score.
**analogical_task.py** calculates score in analogical reasoning task
### Results

The files generated in a run are stored in *ResultsDirectory* specified in configs file<br> 
The files generated on each run will have unique **key**

The following files will be generated in after training
* embedding file of form **[key].embedding.epoch-[epoch-no].out**
* results file(which have concise information of the run) of form **[key].results.txt**

## Config File

```
[LOCATIONS]
ProjectLocation = .
VocabularySource = data/default.vocab
DataFileNamesSplit = data/default.split.txt
AnalogyDataset = data/questions-words.txt
ResultsDirectory = results
ResultsDirectory = results

[MODEL_PARAMETERS]
Epochs = 3
LearningRate = 0.5
EmbeddingDimensions = 60
NegativeSamples = 120
WindowSize = 5
```

## How to Run

create a python3 virtual environment.<br>
In the virtual environment run following to install required packages

```bash
pip3 install -r requirements.txt
```

### Run on available dataset
verify that files in config are present and run following commands from project folder
```
python initialize.py
python driver.py
``` 

### Generate new training split
*filename1* - holds file ids split for training validation and test<br>
*filename2* - has vocabulary generated from train set.
```
python initialize.py filename1 filename2
```

update file names in *config.ini* as following
```
VocabularySource = filename2
DataFileNamesSplit = filename1
```

and run
```
python driver.py
```


## Evaluation

During training and testing is done the embeddings at each epoch are written into file. The files will be *results* folder and file names will be displayed as part of console output.

### Finding k similar words

*param1* - embedding file<br>
*param2* - token<br>
*param3* - output count

```
python similar_words.py embedding.out oil 10
```

### Co-relation score

*param1* - embedding file<br>
*param2* - threshold (minimum occurrences in training set) -optional
```
python simlex_results.py results/emb.out 100
```

### Analytical Reasoning Task

*param1* - embedding file<br>
*param2* - threshold (minimum occurrences in training set) -optional
```
python analogical_task.py results/emb.out 50

```