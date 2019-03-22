from config_handler import Config
from encoder import Encoder
from decoder import Decoder
from training_handler import Trainer
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

    source_vocab, source_size = load_vocabulary(config.source_vocab)
    destination_vocab, destination_size = load_vocabulary(config.destination_vocab)

    encoder = Encoder(source_size, config)
    decoder = Decoder(destination_size, config)

    trainer = Trainer(source_vocab, destination_vocab, encoder, decoder, key=key)
    logs = []

    for epoch in range(config.epochs):
        # train model
        loss, _, batches = trainer.run(config.training_data, config.batch_size, 'train')
        log = "Epoch-" + str(epoch) + " - Avg. Training Loss:" + str(loss/batches)
        print(log)
        logs.append(log)

        # eval model
        loss, score, batches = trainer.run(config.dev_data, config.batch_size, 'dev')
        log = "Epoch-" + str(epoch) + " - Avg. Training Loss:" + str(loss/batches) + " Avg. Score:" + str(score/batches)
        print(log)
        logs.append(log)

    loss, score, batches, count = trainer.run(config.test_data, config.batch_size, 'test')
    log = "Avg. Test Loss:" + str(loss / batches) + " Avg. Score:" + str(score / batches)
    logs.append(log)

    # save model to key.model file
    torch.save(encoder, key + '.encoder')
    torch.save(decoder, key + '.decoder')

    f = open(key + '.result', 'w')
    pp.pprint(config.__dict__, f)
    f.write("\n\n")
    f.write("\n".join(logs))
    pp.pprint(config.__dict__)



