from config_handler import Config
from encoder import Encoder
from decoder import Decoder
from training_handler import Trainer
from datetime import datetime as dt
import random
import string
from utils import *
import torch
import pprint


if __name__ == '__main__':
    config = Config('./config.ini')

    key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))  # generate random string
    print("Key - " + key)

    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(config.__dict__)

    source_size = get_vocab_size(config.source_vocab)
    print("source vocab size - " + str(source_size))
    destination_size = get_vocab_size(config.destination_vocab)
    print("dest vocab size - " + str(destination_size))

    encoder = Encoder(source_size, config)
    decoder = Decoder(destination_size, config)

    trainer = Trainer(encoder.cuda(), decoder.cuda(), config.result_folder + "/" + key)
    logs = []

    for epoch in range(config.epochs):
        # train model
        loss, _, batches = trainer.run(config.training_data, config.batch_size, config.max_batches, 'train')
        log = str(dt.now()) + ": >>>Epoch-" + str(epoch) + " - Avg. Training Loss:" + str(loss/batches)
        print(log)
        logs.append(log)

        # eval model
        loss, _, batches = trainer.run(config.dev_data, config.batch_size, config.max_batches, 'dev')
        log = str(dt.now()) + ": >>>Epoch-" + str(epoch) + " - Avg. Dev Loss:" + str(loss/batches)
        print(log)
        logs.append(log)

        # test model
        loss, score, batches = trainer.run(config.test_data, config.batch_size, config.max_batches, 'test')
        log = str(dt.now()) + ": Avg. Score:" + str(score / batches)
        print(log)
        logs.append(log)

        torch.save(encoder, config.result_folder + "/" + key + '.' + str(epoch) + '.encoder')
        torch.save(decoder, config.result_folder + "/" + key + '.' + str(epoch) + '.decoder')

    f = open(config.result_folder + "/" + key + '.result', 'w')
    f.write(pp.pformat(config.__dict__))
    f.write("\n\n")
    f.write("\n".join(logs))
    pp.pprint(config.__dict__)



