import argparse
from attrdict import AttrDict
from model.utils import openai_transformer_config
import glob
import hashlib
from shutil import copyfile
import pathlib
import datetime
import pprint
from tensorboardX import SummaryWriter
import sys
import random

import logging

logFormatter = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(format=logFormatter, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser()
parser.add_argument("-s", "--signature", default="default_model")
parser.add_argument("-t", "--tie_weights", default=True, type=str2bool)
parser.add_argument("-v", "--path2bert_vocab", default="./supply/rubert_cased_L-12_H-768_A-12/std_lm_vocab.40k.txt")
parser.add_argument("-p", "--tf_bert_model_load_from", default="./supply/rubert_cased_L-12_H-768_A-12/bert_model.ckpt")
parser.add_argument("-o", "--openai_gpt_model_load_from", default="")
parser.add_argument("--openai_bpe_vocab", default=False, type=str2bool)
parser.add_argument("--bare_model", default=False, type=str2bool)
parser.add_argument("--segment_embedding", default=False, type=str2bool)
parser.add_argument("-l", "--load_last", default=True, type=str2bool)
parser.add_argument("--n_layers", default=12, type=int)
parser.add_argument("-b", "--bert_mode", default=True, type=str2bool)
parser.add_argument("-w", "--lr_warmup", default=16000, type=int)
parser.add_argument("-f", "--lr_freq", default=0, type=float)
parser.add_argument("-m", "--lm_weight", default=0.5, type=float)
parser.add_argument("-r", "--risk_weight", default=0, type=float)
parser.add_argument("-d", "--device", default="cuda")
parser.add_argument("--train_from", default="./datasets/toloka_dials/*.train.txt")
parser.add_argument("--valid_from", default="./datasets/toloka_dials/*.valid.txt")
parser.add_argument("--batch_split", default=256, type=int)
parser.add_argument("--input_token_mode", default='default', type=str)
parser.add_argument("--n_epochs", default=1000, type=int)
parser.add_argument("--max_seq_len", default=128, type=int)
# parser.add_argument("--reconf_train_mode", default='default', type=str)
parser.add_argument("--spec_token_reinit", default='', type=str,help='B - init BOSs of SEP, E - init EOSs of CLS') 
args = parser.parse_args()


handler = logging.FileHandler(
    f'logs/{str(args.signature)+ "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")}.txt'
)
handler.setLevel(logging.DEBUG)

# create a logging format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
# add the handlers to the logger
logging.getLogger("").addHandler(handler)

logger.info(" ".join(["python"] + sys.argv))
logger.info("Variable config: {}".format(pprint.pformat(args)))
writer = SummaryWriter(f"tensorboards/{str(args.signature)}")


def get_model_config():
    default_config = openai_transformer_config()
    config = AttrDict(
        {
            "bpe_vocab_path": "./parameters/bpe.vocab",
            "bpe_codes_path": "./parameters/bpe.code",
            "openai_bpe_vocab": args.openai_bpe_vocab,
            "checkpoint_path": "./checkpoints/last_checkpoint",
            "n_layers": args.n_layers,
            "n_pos_embeddings": 512,
            "embeddings_size": default_config.embeddings_size,
            "n_heads": default_config.n_heads,
            "dropout": default_config.dropout,
            "embed_dropout": default_config.embed_dropout,
            "attn_dropout": default_config.attn_dropout,
            "ff_dropout": default_config.ff_dropout,
            "max_seq_len": args.max_seq_len,
            "sep_id_enable": args.bert_mode,
            "beam_size": 3,
            "diversity_coef": 0,
            "diversity_groups": 1,
            "annealing_topk": None,
            "annealing": 0,
            "length_penalty": 0.6,
            "n_segments": args.segment_embedding,
            "bert_mode": args.bert_mode,
            "type_vocab_size": 4,
            "tie_weights": args.tie_weights,
        }
    )
    return config


def get_trainer_config():
    train_files = glob.glob(args.train_from)
    train_hash = hashlib.md5(str(train_files).encode()).hexdigest()[:4]
    random.shuffle(train_files)
    test_files = glob.glob(args.valid_from)
    test_hash = hashlib.md5(str(test_files).encode()).hexdigest()[:4]
    config = AttrDict(
        {
            "n_epochs": args.n_epochs,
            "batch_size": 256,
            "batch_split": args.batch_split,
            "lr": 6.25e-5,
            "lr_warmup": args.lr_warmup,
            "lr_freq": args.lr_freq,
            "lm_weight": args.lm_weight,
            "risk_weight": args.risk_weight,
            "n_jobs": 40,
            "label_smoothing": 0.1,
            "clip_grad": None,
            "test_period": 1,
            "seed": 31415,
            "device": args.device,
            "load_last": args.load_last,
            "bare_model": args.bare_model,
            "tf_bert_model_load_from": args.tf_bert_model_load_from,
            "openai_gpt_model_load_from": args.openai_gpt_model_load_from,
            "bert_vocab": args.path2bert_vocab,
            "openai_parameters_dir": "./parameters",
            "last_checkpoint_path": "./checkpoints/last_checkpoint",
            "interrupt_checkpoint_path": "./checkpoints/interrupt_checkpoint",
            "spec_token_reinit": args.spec_token_reinit,
            "train_datasets": train_files,
            "train_datasets_cache": f"./datasets/train-{train_hash}.cache",
            "test_datasets": test_files,
            "test_datasets_cache": f"./datasets/test-{test_hash}.cache",
            "tb_writer": writer,
            "input_token_mode": args.input_token_mode,
        }
    )
    return config


# %%
def get_config():
    model_config = get_model_config()
    trainer_config = get_trainer_config()
    signature = args.signature

    experiment_path = pathlib.Path("./experiments") / signature
    model_config.checkpoint_path = str(experiment_path / model_config.checkpoint_path)
    trainer_config.last_checkpoint_path = str(experiment_path / trainer_config.last_checkpoint_path)
    trainer_config.load_last = trainer_config.load_last and pathlib.Path(trainer_config.last_checkpoint_path).is_file()

    trainer_config.interrupt_checkpoint_path = str(experiment_path / trainer_config.interrupt_checkpoint_path)
    (experiment_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    return model_config, trainer_config
