#  transformer_chatbot
#  Copyright (C) 2018 Golovanov, Tselousov
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.

import random
import torch
from torch.utils.data import Dataset
from .text import BPEVocab
import traceback
from multiprocessing import Pool
import tqdm

import logging
logger = logging.getLogger(__name__)


class FacebookDataset(Dataset):
    @staticmethod
    def parse_data(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = []
            for line in file.readlines():
                line = line.strip()

                if len(line) == 0:
                    continue

                space_idx = line.find(' ')
                if space_idx == -1:
                    dialog_idx = int(line)
                else:
                    dialog_idx = int(line[:space_idx])

                if int(dialog_idx) == 1:
                    data.append({'persona_info': [], 'dialog': []})

                dialog_line = line[space_idx + 1:].split('\t')
                dialog_line = [l.strip() for l in dialog_line]

                if dialog_line[0].startswith('your persona:'):
                    persona_info = dialog_line[0].replace('your persona: ', '')
                    data[-1]['persona_info'].append(persona_info)

                elif len(dialog_line) > 1:
                    data[-1]['dialog'].append(dialog_line[0])
                    data[-1]['dialog'].append(dialog_line[1])

            return data

    @staticmethod
    def make_dataset(data, vocab, max_lengths):
        dataset = []
        for chat in data:
            persona_info = [vocab.string2ids(s) for s in chat['persona_info']]
            dialog = [vocab.string2ids(s) for s in chat['dialog']]

            if len(dialog) % 2 == 1:
                dialog = dialog[:-1]

            dataset.append((persona_info, dialog))

        return dataset

    @staticmethod
    def run_pool(files, worker, cpu_n=1):
        # Using initializer and  multi_preprocessing functions from this module
        fin_data = []
        with Pool(cpu_n) as p:
            for process_ret in tqdm.tqdm(p.imap_unordered(worker, files), total=len(files)):
                if process_ret:
                    fin_data.append(process_ret)
        return fin_data

    def __init__(self, paths, vocab, max_lengths=2048, min_infos=2, sep_id_enable=False, cpu_n=4):
        assert min_infos > 0

        if isinstance(paths, str):
            paths = [paths]

        self.vocab = vocab
        self.max_lengths = max_lengths
        self.min_infos = min_infos
        self.sep_id_enable = sep_id_enable

        parsed_data = sum(FacebookDataset.run_pool(paths, FacebookDataset.parse_data, cpu_n), [])
        self.data = FacebookDataset.make_dataset(parsed_data, vocab, max_lengths)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        persona_info, dialog = self.data[idx]
        if len(persona_info):
            n_info_samples = max(self.min_infos, random.randint(1, len(persona_info)))
            n_info_samples = min(n_info_samples, len(persona_info))
            persona_info = random.sample(persona_info, n_info_samples)
            random.shuffle(persona_info)
            persona_info = sum(persona_info, [])
            persona_info = [self.vocab.info_bos_id] + persona_info[:self.max_lengths-2] + [self.vocab.info_eos_id]

        dialog_begin = 0
        try:
            dialog_end = random.randrange(2, len(dialog)+1, 2)
        except:
            logger.warning(traceback.format_exc())
            logger.warning(dialog)
            dialog_end = 0

        h = []
        if self.sep_id_enable and persona_info:
            h.append(self.vocab.sep_id)
        for i, ids in enumerate(dialog[dialog_begin:dialog_end-1], 1):
            if i % 2 == 1:
                ids = [self.vocab.talker1_bos_id] + ids + [self.vocab.talker1_eos_id]
            else:
                ids = [self.vocab.talker2_bos_id] + ids + [self.vocab.talker2_eos_id]
            if self.sep_id_enable and len(h) > 1:
                h.append(self.vocab.sep_id)
            h.extend(ids)
        h = h[-self.max_lengths:]

        try:
            y = [self.vocab.bos_id] + dialog[dialog_end-1] + [self.vocab.eos_id]
        except:
            logger.warning(traceback.format_exc())
            logger.warning(dialog)
            y = [self.vocab.bos_id] + [self.vocab.eos_id]
        y = y[:self.max_lengths]

        return persona_info, h, y


# # %%

# import traceback

# def run_user_code():
#     try:
#         assert False
#     except:
#         print("Exception in user code:")
#         print('-'*60)
#         print(traceback.format_exc())
#         print('-'*60)

# run_user_code()
