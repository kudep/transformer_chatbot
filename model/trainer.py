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

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from .utils import pad_sequence
from .optim import Adam, TriangleOpt
from .loss import LabelSmoothingLoss
import numpy as np
import time


from tensorboardX import SummaryWriter
import logging

logger = logging.getLogger(__name__)
tqdm.monitor_interval = 0


class Trainer:
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset=None,
        batch_size=8,
        batch_split=1,
        lm_weight=0.5,
        risk_weight=0,
        lr=6.25e-5,
        lr_warmup=2000,
        lr_freq=0,
        n_jobs=0,
        clip_grad=None,
        label_smoothing=0,
        device=torch.device("cuda"),
        ignore_idxs=[],
        tb_writer=None,
    ):
        self.model = model.to(device)
        self.lm_criterion = nn.CrossEntropyLoss(ignore_index=self.model.padding_idx).to(device)
        self.criterion = LabelSmoothingLoss(
            n_labels=self.model.n_embeddings, smoothing=label_smoothing, ignore_index=self.model.padding_idx
        ).to(device)
        base_optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=0.01)
        step_period = int(int(len(train_dataset) / batch_size) / lr_freq) if lr_freq else float("inf")
        self.optimizer = TriangleOpt(self.model.embeddings_size, 1, lr_warmup, base_optimizer, step_period)

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size // batch_split,
            shuffle=True,
            num_workers=n_jobs,
            collate_fn=self.collate_func,
        )
        if test_dataset is not None:
            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=batch_size // batch_split,
                shuffle=False,
                num_workers=n_jobs,
                collate_fn=self.collate_func,
            )

        self.batch_split = batch_split
        self.lm_weight = lm_weight
        self.risk_weight = risk_weight
        self.clip_grad = clip_grad
        self.device = device
        self.ignore_idxs = ignore_idxs
        self.tb_writer = tb_writer if tb_writer is not None else SummaryWriter("tensorboards")
        self.optimizer._step = 0
        self.epoch = 0

    def state_dict(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "epoch": self.epoch}

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict["model"], strict=False)
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.epoch = state_dict.get("epoch", 0)

    def collate_func(self, data):
        persona_info, h, y = zip(*data)

        contexts = []

        if max(map(len, persona_info)) > 0:
            persona_info = [torch.tensor(d, dtype=torch.long) for d in persona_info]
            persona_info = pad_sequence(persona_info, batch_first=True, padding_value=self.model.padding_idx)
            contexts.append(persona_info)

        if max(map(len, h)) > 0:
            h = [torch.tensor(d, dtype=torch.long) for d in h]
            h = pad_sequence(h, batch_first=True, padding_value=self.model.padding_idx)
            contexts.append(h)

        y = [torch.tensor(d, dtype=torch.long) for d in y]
        y = pad_sequence(y, batch_first=True, padding_value=self.model.padding_idx)

        return contexts, y

    def _eval_train(self, epoch, risk_func=None):
        self.model.train()

        tqdm_data = tqdm(self.train_dataloader, desc="Train (epoch #{})".format(epoch))
        loss = 0
        lm_loss = 0
        risk_loss = 0
        for i, (contexts, targets) in enumerate(tqdm_data):
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)

            enc_contexts = []

            # lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            for context in contexts:
                enc_context = self.model.encode(context.clone())
                enc_contexts.append(enc_context)

                if self.lm_weight > 0:
                    context_outputs = self.model.generate(enc_context[0])
                    ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                    context.masked_fill_(ignore_mask, self.model.padding_idx)
                    prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                    batch_lm_loss += self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / len(contexts)

            # s2s loss
            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            # risk loss
            batch_risk_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            if risk_func is not None and self.risk_weight > 0:

                beams, beam_lens = self.model.beam_search(enc_contexts, return_beams=True)

                target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
                targets = [target[1 : length - 1].tolist() for target, length in zip(targets, target_lens)]
                batch_risks = []
                for b in range(beams.shape[1]):
                    predictions = [b[1 : l - 1].tolist() for b, l in zip(beams[:, b, :], beam_lens[:, b])]
                    risks = torch.tensor(risk_func(predictions, targets), dtype=torch.float, device=self.device)
                    batch_risks.append(risks)
                batch_risks = torch.stack(batch_risks, dim=-1)

                batch_probas = []
                for b in range(beams.shape[1]):
                    logits = self.model.decode(beams[:, b, :-1], enc_contexts)
                    probas = F.log_softmax(logits, dim=-1)
                    probas = torch.gather(probas, -1, beams[:, b, 1:].unsqueeze(-1)).squeeze(-1)
                    probas = probas.sum(dim=-1) / beam_lens[:, b].float()
                    batch_probas.append(probas)
                batch_probas = torch.stack(batch_probas, dim=-1)
                batch_probas = F.softmax(batch_probas, dim=-1)

                batch_risk_loss = torch.mean((batch_risks * batch_probas).sum(dim=-1))

            # optimization
            full_loss = (
                batch_lm_loss * self.lm_weight + self.risk_weight * batch_risk_loss + batch_loss
            ) / self.batch_split
            full_loss.backward()

            if (i + 1) % self.batch_split == 0:
                if self.clip_grad is not None:
                    for group in self.optimizer.param_groups:
                        nn.utils.clip_grad_norm_(group["params"], self.clip_grad)
                self.optimizer.step()
                self._tb_writer_after_step()
                self.optimizer.zero_grad()

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            risk_loss = (i * risk_loss + batch_risk_loss.item()) / (i + 1)

            tqdm_data.set_postfix({"lm_loss": lm_loss, "loss": loss, "risk_loss": risk_loss})
            if (i + 1) % self.batch_split == 0:
                self.tb_writer.add_scalar("train/lm_loss", lm_loss, self.optimizer._step)
                self.tb_writer.add_scalar("train/risk_loss", risk_loss, self.optimizer._step)
                self.tb_writer.add_scalar("train/loss", loss, self.optimizer._step)

    def _mean4hist(self, component):
        return component.weight.grad.clone().cpu().data.numpy()

    def _tb_writer_after_step(self):
        self.tb_writer.add_scalar("optimizer/rate", self.optimizer.rate(), self.optimizer._step)
        if self.optimizer._step % 50 == 1:  # it's take 16 seconds.
            tensors = []
            embeddings = self._mean4hist(self.model.transformer_module.embeddings)
            tensors.append(embeddings)
            self.tb_writer.add_histogram("optimizer/grads/embeddings", embeddings, self.optimizer._step)
            if self.model.transformer_module.n_segments:
                type_embeddings = self._mean4hist(self.model.transformer_module.type_embeddings)
                tensors.append(type_embeddings)
                self.tb_writer.add_histogram("optimizer/grads/type_embeddings", type_embeddings, self.optimizer._step)
            pos_embeddings = self._mean4hist(self.model.transformer_module.pos_embeddings)
            tensors.append(pos_embeddings)
            self.tb_writer.add_histogram("optimizer/grads/pos_embeddings", pos_embeddings, self.optimizer._step)
            pre_softmax = self._mean4hist(self.model.pre_softmax)
            tensors.append(pre_softmax)
            self.tb_writer.add_histogram("optimizer/grads/pre_softmax", pre_softmax, self.optimizer._step)
            for i, layer in enumerate(self.model.transformer_module.layers, 1):
                layer_grads = [tensor.grad.clone().cpu().data.numpy() for tensor in layer.parameters()]
                layer_grads = [tensor.reshape(-1) for tensor in layer_grads]
                layer_grads = np.concatenate(layer_grads)
                tensors.append(layer_grads)
                self.tb_writer.add_histogram(f"optimizer/grads/layer_{i:02d}", layer_grads, self.optimizer._step)
            tensors = [tensor.reshape(-1) for tensor in tensors]
            tensors = np.concatenate(tensors)
            self.tb_writer.add_histogram(f"optimizer/grads/global", tensors, self.optimizer._step)

    def _get_timeout(self, start=False, paticient=10, timeout=2):
        if start:
            self.base_paticient = paticient
            self.paticient = paticient
            self.timeout = timeout
            self.last = time.time()
        current = time.time()
        if timeout > (current - self.last):
            self.paticient = min(self.paticient + 1, self.base_paticient)
        if timeout <= (current - self.last):
            self.paticient -= 1
        self.last = time.time()
        return self.paticient <= 0

    def _eval_test(self, metric_funcs={}):
        self.model.eval()

        tqdm_data = tqdm(self.test_dataloader, desc="Test")
        loss = 0
        lm_loss = 0
        metrics = {name: 0 for name in metric_funcs.keys()}
        self._get_timeout(True)

        for i, (contexts, targets) in enumerate(tqdm_data):
            if self._get_timeout():
                logger.info(f"Timeout on step: {i}")
                break
            contexts, targets = [c.to(self.device) for c in contexts], targets.to(self.device)

            enc_contexts = []

            # lm loss
            batch_lm_loss = torch.tensor(0, dtype=torch.float, device=self.device)
            for context in contexts:
                enc_context = self.model.encode(context.clone())
                enc_contexts.append(enc_context)

                if self.lm_weight > 0:
                    context_outputs = self.model.generate(enc_context[0])
                    ignore_mask = torch.stack([context == idx for idx in self.ignore_idxs], dim=-1).any(dim=-1)
                    context.masked_fill_(ignore_mask, self.model.padding_idx)
                    prevs, nexts = context_outputs[:, :-1, :].contiguous(), context[:, 1:].contiguous()
                    batch_lm_loss += self.lm_criterion(prevs.view(-1, prevs.shape[-1]), nexts.view(-1)) / len(contexts)

            # s2s loss
            prevs, nexts = targets[:, :-1].contiguous(), targets[:, 1:].contiguous()
            outputs = self.model.decode(prevs, enc_contexts)
            outputs = F.log_softmax(outputs, dim=-1)
            batch_loss = self.criterion(outputs.view(-1, outputs.shape[-1]), nexts.view(-1))

            predictions = self.model.beam_search(enc_contexts)
            target_lens = targets.ne(self.model.padding_idx).sum(dim=-1)
            targets = [t[1 : l - 1].tolist() for t, l in zip(targets, target_lens)]

            lm_loss = (i * lm_loss + batch_lm_loss.item()) / (i + 1)
            loss = (i * loss + batch_loss.item()) / (i + 1)
            for name, func in metric_funcs.items():
                score = func(predictions, targets)
                if score is not None:
                    metrics[name] = (metrics[name] * i + score) / (i + 1)

            tqdm_data.set_postfix(dict({"lm_loss": lm_loss, "loss": loss}, **metrics))

        self.tb_writer.add_scalar("test/lm_loss", lm_loss, self.epoch)
        self.tb_writer.add_scalar("test/loss", loss, self.epoch)
        for k, v in metrics.items():
            self.tb_writer.add_scalar(f"test/{k}", v, self.epoch)
        logger.info(f"Test: {dict({'lm_loss': lm_loss, 'loss': loss}, **metrics)}")

    def test(self, metric_funcs={}):
        if hasattr(self, "test_dataloader"):
            self._eval_test(metric_funcs)

    def train(self, epochs, after_epoch_funcs=[], risk_func=None):
        for _ in range(epochs):
            self._eval_train(self.epoch, risk_func)
            logger.info(f"End of epoch # {self.epoch}")
            self.epoch += 1

            for func in after_epoch_funcs:
                func(self.epoch)
