import configparser


class Config:

    def __init__(self, file=None):
        # parse the config file
        config = configparser.ConfigParser()
        if file is None:
            file = 'config.ini'
        config.read(file)

        self.source_data = config.get('DATA', 'SourceData')
        self.destination_data = config.get('DATA', 'DestinationData')
        self.source_dev_data = config.get('DATA', 'SourceDevData')
        self.destination_dev_data = config.get('DATA', 'DestinationDevData')
        self.source_test_data = config.get('DATA', 'SourceTestData')
        self.destination_test_data = config.get('DATA', 'DestinationTestData')
        self.max_sent_length = config.getint('DATA', 'MaxSentenceLength')

        self.processed_training_data = config.get('DATA', 'ProcessedTrainingData')
        self.processed_dev_data = config.get('DATA', 'ProcessedDevData')
        self.processed_test_data = config.get('DATA', 'ProcessedTestData')

        self.training_data = config.get('DATA', 'TrainingData')
        self.dev_data = config.get('DATA', 'DevData')
        self.test_data = config.get('DATA', 'TestData')

        self.source_lang = config.get('DATA', 'SourceLang')
        self.destination_lang = config.get('DATA', 'DestinationLang')

        self.source_vocab = config.get('DATA', 'SourceVocab')
        self.destination_vocab = config.get('DATA', 'DestinationVocab')
        self.min_freq = config.getint('DATA', 'MinimumFrequency')

        self.result_folder = config.get('DATA', 'ResultDir')
        self.checkpoint_folder = config.get('DATA', 'CheckpointDir')

        self.encoder_embedding_size = config.getint('MODEL', 'SourceEmbeddingDim')
        self.decoder_embedding_size = config.getint('MODEL', 'DestinationEmbeddingDim')
        self.layers = config.getint('MODEL', 'LSTMLayers')
        self.learning_rate = config.getfloat('MODEL', 'LearningRate')
        self.hidden_units = config.getint('MODEL', 'LSTMHiddenUnits')

        params = {
            "encoder_attn": config.get('ATTENTION', 'EncoderAttention'),
            "decoder_attn": config.get('ATTENTION', 'DecoderAttention'),
            "self_attn": config.getboolean('ATTENTION', 'SelfAttention'),
            "heads": config.getint('ATTENTION', 'AttentionHeads')
        }

        kv_split = config.get('ATTENTION', 'KeyValueSplit').split(",")
        if len(kv_split) != 2:
            kv_split = None
        else:
            kv_split = [int(i) for i in kv_split]
        params["key_value_split"] = kv_split

        self.attention_params = params

        self.epochs = config.getint('TRAINING', 'Epochs')

        self.training_batch_size = config.getint('TRAINING', 'TrainingBatchSize')
        self.max_training_batches = config.getint('TRAINING', 'MaxTrainingBatches')

        self.dev_batch_size = config.getint('TRAINING', 'DevBatchSize')
        self.max_dev_batches = config.getint('TRAINING', 'MaxDevBatches')

        self.test_batch_size = config.getint('TRAINING', 'TestBatchSize')
        self.max_test_batches = config.getint('TRAINING', 'MaxTestBatches')
