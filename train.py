import torch
import random
from model.utils import load_openai_weights, set_seed, f1_score
from model.transformer_model import TransformerModel
from model.trainer import Trainer
from model.text import BPEVocab, BertBPEVocab
from model.dataset import FacebookDataset
from config import get_model_config, get_trainer_config


import tensorflow as tf
import re
import numpy as np


def load_from_bert(model, n_embeddings, model_config):
    # for name, W in model.named_parameters():
    #     print(f'name {name} shape {W.shape}')
    #     W = torch.full_like(W, 0)

    tf_path = "/home/den/Documents/chit-chat_2019/models/bert_models/multi_cased_L-12_H-768_A-12/bert_model.ckpt"
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        # print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)
    for name, array in zip(names, arrays):
        origin_name = name

        name = name.replace('bert/embeddings/word_embeddings', 'embeddings')
        name = name.replace('bert/embeddings/LayerNorm', 'embed_norm')
        name = name.replace('bert/embeddings/position_embeddings', 'pos_embeddings')
        name = name.replace('bert/embeddings/token_type_embeddings', 'type_embeddings')

        name = name.replace('bert/encoder/layer', 'layers')
        name = name.replace('attention/output/LayerNorm', 'attn_norm')
        name = name.replace('attention', 'attn')
        name = name.replace('attn/output/dense', 'attn/out_proj')
        name = name.replace('kernel', 'weight')
        name = name.replace('gamma', 'weight')
        name = name.replace('beta', 'bias')
        name = name.replace('output_bias', 'bias')
        name = name.replace('output_weights', 'weight')
        name = name.replace('key/bias', 'self_attn_key_bias')
        name = name.replace('key/weight', 'self_attn_key_weight')
        name = name.replace('query/bias', 'self_attn_query_bias')
        name = name.replace('query/weight', 'self_attn_query_weight')
        name = name.replace('value/bias', 'self_attn_value_bias')
        name = name.replace('value/weight', 'self_attn_value_weight')
        name = name.replace('self/', '')
        name = name.replace('intermediate/dense', 'ff/layer_1')
        name = name.replace('output/dense', 'ff/layer_2')
        name = name.replace('output/LayerNorm', 'ff_norm')

        splitted_name = name.split('/')

        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step", "cls", "bert"] for n in splitted_name):
            print("Skipping {}".format(origin_name))
            continue

        pointer = model
        for m_name in splitted_name:
            if re.fullmatch(r'layers_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            try:
                if 'self_attn_' in l[0]:
                    pointer = getattr(pointer, 'qkv_proj')
                    l[0] = l[0].split('_')[-1]  # for self_attn_(key/query/value)_(weight/bias)
                pointer = getattr(pointer, l[0])
                if len(l) >= 2:
                    num = int(l[1])
                    pointer = pointer[num]
            except Exception as ex:
                print(ex)
                print(m_name)
                print(l)
                print(name)
                print(origin_name)
                print(f'array {array.shape}')
                return

        if 'kernel' in origin_name:
            array = np.transpose(array)
        if 'embeddings' in name:
            pointer = getattr(pointer, 'weight')

        if name == 'embeddings':
            array = array[:n_embeddings]  # slicing of embeddings
        if name == 'type_embeddings':
            mean_type_emb = array.mean(axis=0)
            new_array = np.stack([mean_type_emb]*model_config.type_vocab_size)
            start_index = model_config.type_vocab_size//2 - array.shape[0]//2
            new_array[start_index:start_index+array.shape[0]] = array
            array = new_array

        if pointer.shape != array.shape and not ('self_attn_' in name):
            print(m_name)
            print(l)
            print(name)
            print(origin_name)
            print(f'pointer {pointer.shape}')
            print(f'array {array.shape}')
            assert False
        if 'self_attn_' in name:
            if 'query' in name:
                shift_index = 0
            elif 'key' in name:
                shift_index = 1
            elif 'value' in name:
                shift_index = 2
            else:
                assert False
            dim1 = array.shape[0]
            pointer.data[dim1*(shift_index):dim1*(shift_index+1)] = torch.from_numpy(array)
        else:
            pointer.data = torch.from_numpy(array)


def main():
    model_config = get_model_config()
    trainer_config = get_trainer_config()

    set_seed(trainer_config.seed)
    device = torch.device(trainer_config.device)

    if trainer_config.first_load_from_tf_bert:
        path = '/home/den/Documents/chit-chat_2019/models/bert_models/rubert_cased_L-12_H-768_A-12/std_lm_vocab.txt'
        path = '/home/den/Documents/chit-chat_2019/models/bert_models/rubert_cased_L-12_H-768_A-12/std_lm_vocab.40k.txt'
        vocab = BertBPEVocab.from_files(path)
    else:
        vocab = BPEVocab.from_files(model_config.bpe_vocab_path, model_config.bpe_codes_path)

    transformer = TransformerModel(n_layers=model_config.n_layers,
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

    if trainer_config.first_load_from_tf_bert:
        load_from_bert(transformer.transformer_module, len(vocab), model_config)

    # if not trainer_config.load_last:
    #     load_openai_weights(transformer.transformer_module,
    #                         trainer_config.openai_parameters_dir,
    #                         n_special_tokens=vocab.n_special_tokens)
    #     print('OpenAI weights loaded from {}'.format(trainer_config.openai_parameters_dir))

    test_dataset = FacebookDataset(trainer_config.test_datasets,
                                   vocab,
                                   transformer.n_pos_embeddings - 1,
                                   sep_id_enable=model_config.sep_id_enable
                                   )

    # train_dataset = FacebookDataset(trainer_config.train_datasets,
    #                                vocab,
    #                                transformer.n_pos_embeddings - 1,
    #                                sep_id_enable=model_config.sep_id_enable
    #                                )

    train_dataset = test_dataset

    model_trainer = Trainer(transformer,
                            train_dataset,
                            test_dataset,
                            batch_size=trainer_config.batch_size,
                            batch_split=trainer_config.batch_split,
                            lr=trainer_config.lr,
                            lr_warmup=trainer_config.lr_warmup,
                            lm_weight=trainer_config.lm_weight,
                            risk_weight=trainer_config.risk_weight,
                            n_jobs=trainer_config.n_jobs,
                            clip_grad=trainer_config.clip_grad,
                            device=device,
                            ignore_idxs=vocab.special_tokens_ids)

    # if trainer_config.load_last:
    #     state_dict = torch.load(trainer_config.last_checkpoint_path, map_location=device)
    #     model_trainer.load_state_dict(state_dict)
    #     print('Weights loaded from {}'.format(trainer_config.last_checkpoint_path))

    # helpers -----------------------------------------------------

    def save_func(epoch):
        torch.save(model_trainer.state_dict(), trainer_config.last_checkpoint_path)

    def sample_text_func(epoch):
        n_samples = 5
        samples_idxs = random.sample(range(len(test_dataset)), n_samples)
        samples = [test_dataset[idx] for idx in samples_idxs]
        for persona_info, dialog, target in samples:
            contexts = [torch.tensor([c], dtype=torch.long, device=model_trainer.device) for c in [persona_info, dialog] if len(c) > 0]
            prediction = model_trainer.model.predict(contexts)[0]

            persona_info_str = vocab.ids2string(persona_info[1:-1])
            dialog_str = vocab.ids2string(dialog)
            dialog_str = dialog_str.replace(vocab.talker1_bos, '\n\t- ').replace(vocab.talker2_bos, '\n\t- ')
            dialog_str = dialog_str.replace(vocab.talker1_eos, '').replace(vocab.talker2_eos, '')
            if model_config.sep_id_enable:
                dialog_str = dialog_str.replace(vocab.sep_token, '')
            target_str = vocab.ids2string(target[1:-1])
            prediction_str = vocab.ids2string(prediction)

            print('\n')
            print('Persona info:\n\t{}'.format(persona_info_str))
            print('Dialog:{}'.format(dialog_str))
            print('Target:\n\t{}'.format(target_str))
            print('Prediction:\n\t{}'.format(prediction_str))

    def test_func(epoch):
        if (epoch+1) % trainer_config.test_period == 0:
            metric_funcs = {'f1_score': f1_score}
            model_trainer.test(metric_funcs)

    def f1_risk(predictions, targets):
        scores = f1_score(predictions, targets, average=False)
        return [1-s for s in scores]

    # helpers -----------------------------------------------------

    try:
        model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func, sample_text_func], risk_func=f1_risk)
        # model_trainer.train(trainer_config.n_epochs, after_epoch_funcs=[save_func, sample_text_func, test_func], risk_func=f1_risk)
    except (KeyboardInterrupt, Exception, RuntimeError) as e:
        torch.save(model_trainer.state_dict(), trainer_config.interrupt_checkpoint_path)
        raise e


if __name__ == '__main__':
    main()
