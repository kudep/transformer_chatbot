import torch
import random
from model.utils import (
    load_openai_weights,
    set_seed,
    f1_score,
    update_vocab_counters,
    clear_vocab_counters,
    get_vocab_stats,
)
from model.load_bert import load_from_bert
from model.transformer_model import TransformerModel
from model.trainer import Trainer
from model.text import BPEVocab, BertBPEVocab
from model.dataset import FacebookDataset
from config import get_config

import pprint

import logging

logger = logging.getLogger(__name__)


def main():
    model_config, trainer_config = get_config()
    logger.info("Model config: {}".format(pprint.pformat(model_config)))
    logger.info("Trainer config: {}".format(pprint.pformat(trainer_config)))

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    if model_config.openai_bpe_vocab:
        logger.info("Vocab loaded from {}".format(trainer_config.bpe_vocab_path))
        vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)
    else:
        logger.info("Bert bpe vocab  loaded from {}".format(trainer_config.bert_vocab))
        vocab = BertBPEVocab.from_files(trainer_config.bert_vocab)

    transformer = TransformerModel(
        n_layers=model_config.n_layers,
        n_embeddings=len(vocab),
        n_pos_embeddings=model_config.n_pos_embeddings,
        embeddings_size=model_config.embeddings_size,
        padding_idx=vocab.pad_id,
        n_heads=model_config.n_heads,
        dropout=model_config.dropout,
        embed_dropout=model_config.embed_dropout,
        attn_dropout=model_config.attn_dropout,
        ff_dropout=model_config.ff_dropout,
        bos_id=vocab.bos_id,
        eos_id=vocab.eos_id,
        max_seq_len=model_config.max_seq_len,
        beam_size=model_config.beam_size,
        length_penalty=model_config.length_penalty,
        n_segments=model_config.n_segments,
        annealing_topk=model_config.annealing_topk,
        annealing=model_config.annealing,
        diversity_coef=model_config.diversity_coef,
        diversity_groups=model_config.diversity_groups,
        bert_mode=model_config.bert_mode,
        type_vocab_size=model_config.type_vocab_size,
        tie_weights=model_config.tie_weights,
        info_bos_id=vocab.info_bos_id,
        talker1_bos_id=vocab.talker1_bos_id,
        talker2_bos_id=vocab.talker2_bos_id,
        bos_token_id=vocab.bos_id,
        sep_token_id=vocab.sep_id,
    )

    if not (trainer_config.bare_model or trainer_config.load_last):
        if trainer_config.openai_gpt_model_load_from:
            logger.info("OpenAI weights loading from {}".format(trainer_config.openai_parameters_dir))
            load_openai_weights(
                transformer.transformer_module,
                trainer_config.openai_parameters_dir,
                n_special_tokens=vocab.n_special_tokens,
            )
        else:
            logger.info("Weights loading from {}".format(trainer_config.tf_bert_model_load_from))
            load_from_bert(transformer.transformer_module, vocab, model_config, trainer_config)

    logger.info("Test data loading ")
    test_dataset = FacebookDataset(
        trainer_config.test_datasets,
        vocab,
        model_config.max_seq_len,
        sep_id_enable=model_config.sep_id_enable,
        cpu_n=trainer_config.n_jobs,
        cache_file=trainer_config.test_datasets_cache,
        input_token_mode=trainer_config.input_token_mode,
    )

    logger.info("Train data loading ")
    train_dataset = FacebookDataset(
        trainer_config.train_datasets,
        vocab,
        model_config.max_seq_len,
        sep_id_enable=model_config.sep_id_enable,
        cpu_n=trainer_config.n_jobs,
        cache_file=trainer_config.train_datasets_cache,
        input_token_mode=trainer_config.input_token_mode,
    )
    model_trainer = Trainer(
        transformer,
        train_dataset,
        test_dataset,
        batch_size=trainer_config.batch_size,
        batch_split=trainer_config.batch_split,
        lr=trainer_config.lr,
        lr_warmup=trainer_config.lr_warmup,
        lr_freq=trainer_config.lr_freq,
        lm_weight=trainer_config.lm_weight,
        risk_weight=trainer_config.risk_weight,
        n_jobs=trainer_config.n_jobs,
        clip_grad=trainer_config.clip_grad,
        device=device,
        ignore_idxs=vocab.special_tokens_ids,
        tb_writer=trainer_config.tb_writer,
    )

    if trainer_config.load_last:
        logger.info("Weights loading from {}".format(trainer_config.last_checkpoint_path))
        state_dict = torch.load(trainer_config.last_checkpoint_path, map_location=device)
        model_trainer.load_state_dict(state_dict)

    # helpers -----------------------------------------------------

    def save_func(epoch):
        torch.save(
            model_trainer.state_dict(),
            trainer_config.last_checkpoint_path + f"_ep-{epoch}" + f"_st-{model_trainer.optimizer._step}",
        )
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path)

    def sample_text_func(epoch):
        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        # samples = [test_dataset[idx] for idx in samples_idxs]
        samples = [test_dataset[idx] for idx in range(n_samples)]

        for persona_info, dialog, target in samples:
            contexts = [
                torch.tensor([c], dtype=torch.long, device=model_trainer.device)
                for c in [persona_info, dialog]
                if len(c) > 0
            ]
            prediction = model_trainer.model.predict(contexts)[0]

            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, "\n\t- ").replace(vocab.talker2_bos, "\n\t- ")
            dialog_str = dialog_str.replace(vocab.talker1_eos, "").replace(vocab.talker2_eos, "")
            for special_token_id in vocab.special_tokens_ids:
                dialog_str = dialog_str.replace(special_token_id, "")
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            inf_out = (
                "\nDialog:{}".format(dialog_str)
                + "\nTarget:\n\t{}".format(target_str)
                + "\nPrediction: \n\t{}".format(prediction_str)
                + "\nPrediction len: \n\t{}".format(len(prediction))
            )
            if persona_info_str:
                logger.info("\nPersona info:\n\t{}".format(persona_info_str) + inf_out)
            else:
                logger.info(inf_out)

    def test_func(epoch):
        if (epoch + 1) % trainer_config.test_period == 0:
            metric_funcs = {
                "f1_score": f1_score,
                "vocab_stat_powerset_score": update_vocab_counters,
            }
            model_trainer.test(metric_funcs)
            post_metrics = get_vocab_stats()
            clear_vocab_counters()
            for k, v in post_metrics.items():
                trainer_config.tb_writer.add_scalar(f"test/{k}", v, epoch)
            logger.info(f"Test post metrics: {dict({}, **post_metrics)}")

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        return [1 - s for s in scores]

    # helpers -----------------------------------------------------

    try:
        if trainer_config.load_last:
            save_func(model_trainer.epoch)
            sample_text_func(model_trainer.epoch)
            test_func(model_trainer.epoch)

        model_trainer.train(
            trainer_config.n_epochs, after_epoch_funcs=[save_func, sample_text_func, test_func], risk_func=f1_risk
        )

        # model_trainer.train(
        #     trainer_config.n_epochs, after_epoch_funcs=[], risk_func=f1_risk
        # )
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(
            model_trainer.state_dict(),
            trainer_config.interrupt_checkpoint_path + f"_st-{model_trainer.optimizer._step}",
        )
        raise e


if __name__ == "__main__":
    main()