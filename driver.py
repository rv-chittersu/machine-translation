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
import sys


if __name__ == '__main__':
    config = Config('./config.ini')

    # getting vocabulary size
    source_size = get_vocab_size(config.source_vocab)
    print("source vocab size - " + str(source_size))
    destination_size = get_vocab_size(config.destination_vocab)
    print("dest vocab size - " + str(destination_size))

    # generating unique key to track run
    key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))  # generate random string
    init_epoch = -1

    # start new run or reload old run
    if len(sys.argv) == 3:
        key = sys.argv[1]
        init_epoch = int(sys.argv[2])
        print("Loading run - " + key + " at epoch - " + sys.argv[2])
        encoder = torch.load(config.checkpoint_folder + "/" + key + ".encoder")
        decoder = torch.load(config.checkpoint_folder + "/" + key + ".decoder")
        f = open(config.result_folder + "/" + key + ".result", "a+")
    else:
        encoder = Encoder(source_size, config)
        decoder = Decoder(destination_size, config)
        f = open(config.result_folder + "/" + key + '.result', 'w')

    print("Key - " + key)

    # getting reverse vocab
    src_vcb = get_reverse_vocab(config.source_vocab)
    dst_vcb = get_reverse_vocab(config.destination_vocab)

    # creating trainer
    trainer = Trainer(encoder.cuda(), decoder.cuda(), config.result_folder + "/" + key, src_vcb, dst_vcb)

    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(config.__dict__)
    f.write(pp.pformat(config.__dict__))
    f.write("\n\n")

    for epoch in range(init_epoch + 1, config.epochs):
        try:
            # train
            loss, batches = trainer.run(config.processed_training_data, config.training_batch_size, config.max_training_batches, 'train')
            log = str(dt.now()) + ": >>>Epoch-" + str(epoch) + " - Avg. Training Loss:" + str(loss/batches)
            print(log)
            f.write(log + "\n")
            f.flush()

            # validate
            loss, batches = trainer.run(config.processed_dev_data, config.dev_batch_size, config.max_dev_batches, 'dev')
            log = str(dt.now()) + ": >>>Epoch-" + str(epoch) + " - Avg. Dev Loss:" + str(loss/batches)
            print(log)
            f.write(log + "\n")
            f.flush()

            # test
            _, _ = trainer.run(config.processed_test_data, config.test_batch_size, config.max_test_batches, 'test')
            log = str(dt.now()) + ": >>> BLEU Score on test:" + str(compute_bleu_score(config.result_folder + "/" + key))
            print(log)
            f.write(log + "\n")
            f.flush()

        except Exception as e:
            torch.save(encoder, config.checkpoint_folder + "/" + key + '.encoder')
            torch.save(decoder, config.checkpoint_folder + "/" + key + '.decoder')
            raise e

    # save trained encoder and decoder
    torch.save(encoder, config.checkpoint_folder + "/" + key + '.encoder')
    torch.save(decoder, config.checkpoint_folder + "/" + key + '.decoder')

    f.close()
    print("Key - " + key)



