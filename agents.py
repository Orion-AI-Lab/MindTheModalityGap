import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from heads import get_classification_head
from models import ImageClassifier, ImageEncoder, KDClassifier
from utils import cosine_lr, instantiate
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    def __init__(self, cfg):
        self.config = cfg
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def load_checkpoint(self, filename):
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, filename):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def train_one_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def validate(self):
        raise NotImplementedError

    @abstractmethod
    def finalize(self):
        raise NotImplementedError


class PatchingAgent(BaseAgent):
    def __init__(self, cfg, model, tokenizer, task_dataset, eval_dataset):
        super().__init__(cfg)

        self.model = model
        self.tokenizer = tokenizer
        self.task_dataset = task_dataset
        self.eval_dataset = eval_dataset

        self.task_config = lambda setting: self.config["task"][setting]

        self._init_classification_heads()
        self._init_task_model()

        self.scaler = GradScaler()
        self.device = self.config["torch"]["device"]
        self.loss = instantiate(self.config["data"]["task_dataset"]["loss"]).to(
            self.device
        )
        self.optimizer = instantiate(
            self.config["task"]["optimizer"],
            params=[p for p in self.model.parameters() if p.requires_grad],
        )

        self.current_epoch = 0
        self.current_iteration = 0
        self.current_alpha = 0.0

        self.best_metric = 0

    def _init_classification_heads(self):
        for dataset in self.eval_dataset:
            get_classification_head(self.config, self.model, self.tokenizer, dataset)

    def _init_task_model(self):
        self.model = ImageClassifier(
            classification_head=get_classification_head(
                self.config, self.model, self.tokenizer, self.task_dataset["train"]
            ),
            image_encoder=ImageEncoder(self.model, keep_lang=False),
        ).to(self.config["torch"]["device"])
        self.model.freeze_head()
        self.model = (
            torch.compile(self.model, backend="inductor")
            if self.config["torch"]["torch_compile"]
            else self.model
        )

    def load_checkpoint(self, filename):
        raise NotImplementedError

    def save_checkpoint(self):
        prefix = "_orig_mod." if self.config["torch"]["torch_compile"] else ""

        if self.current_iteration:
            suffix = (
                f"paint_{str(self.current_alpha).replace('.', '')}"
                if self.current_alpha
                else f"ft_{self.current_epoch}"
            )
        else:
            suffix = "zs"

        state_dict = self.model.state_dict()

        for key in list(state_dict.keys()):
            if f"{prefix}image_encoder.model." in key:
                state_dict[key.replace(f"{prefix}image_encoder.model.", "")] = (
                    state_dict[key]
                )
                del state_dict[key]
            if "classification_head" in key:
                del state_dict[key]

        ckpt = Path(self.config["env"]["work_dir"]) / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            ckpt
            / f"{self.config['model']['name']}_{self.config['model']['pretrained']}_{suffix}.pt",
        )

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("SIGTERM received, exiting gracefully...")

    def train(self):
        self.save_checkpoint()
        for epoch in range(0, self.config["task"]["num_epochs"]):
            self.train_one_epoch()
            self.save_checkpoint()
            if (epoch + 1) == self.config["task"]["num_epochs"]:
                self.validate(dataset=self.task_dataset["test"])
            elif (epoch + 1) % self.config["task"]["eval_frequency"] == 0:
                self.validate(dataset=self.task_dataset["val"])
        self.patch()

    def train_one_epoch(self):
        self.model.train()
        scheduler = cosine_lr(
            self.optimizer,
            self.config["task"]["optimizer"]["lr"],
            self.config["task"]["lr_scheduler"]["warmup_length"],
            self.task_dataset["train"].num_batches,
        )

        for batch_idx, (images, targets) in enumerate(self.task_dataset["train"]):
            scheduler(
                step=batch_idx
                + self.current_epoch * self.task_dataset["train"].num_batches
            )
            self.optimizer.zero_grad(set_to_none=True)
            images = images.to(
                self.config["torch"]["device"],
                non_blocking=True,
                dtype=torch.bfloat16,
            )
            targets = targets.to(self.config["torch"]["device"], non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model(images)
                loss = self.loss(output, targets)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if batch_idx % self.config["torch"]["log_interval"] == 0:
                self.logger.info(
                    f"Train Epoch: {self.current_epoch} [{batch_idx * len(images)}/{self.task_dataset['train'].num_samples} ({100.0 * batch_idx / self.task_dataset['train'].num_batches:.0f}%)] Loss: {loss.item():.6f} LR: {self.optimizer.param_groups[0]['lr']}"
                )
            self.current_iteration += 1

        if self.current_epoch != self.config["task"]["num_epochs"] - 1:
            self.current_epoch += 1

    def patch(self):
        ckpt_zs = (
            Path(self.config["env"]["work_dir"])
            / "checkpoints"
            / f"{self.config['model']['name']}_{self.config['model']['pretrained']}_zs.pt"
        )
        ckpt_ft = (
            Path(self.config["env"]["work_dir"])
            / "checkpoints"
            / f"{self.config['model']['name']}_{self.config['model']['pretrained']}_ft_{self.current_epoch}.pt"
        )

        if ckpt_zs.exists():
            zeroshot_state_dict = torch.load(ckpt_zs, map_location="cpu")["model"]
            finetuned_state_dict = torch.load(ckpt_ft, map_location="cpu")["model"]
        else:
            self.logger.error("Missing checkpoint(s).")
            raise FileNotFoundError

        theta_0 = {k: v.clone() for k, v in zeroshot_state_dict.items()}
        theta_1 = {k: v.clone() for k, v in finetuned_state_dict.items()}
        assert set(theta_0.keys()) == set(theta_1.keys())

        for alpha in self.config["task"]["alphas"]:
            self.current_alpha = alpha
            self.logger.info("=" * 100)
            self.logger.info(f"Evaluating with alpha={alpha:.2f}")

            # interpolate between all weights in the checkpoints
            theta = {
                key: (1 - self.current_alpha) * theta_0[key]
                + self.current_alpha * theta_1[key]
                for key in theta_0.keys()
            }

            # update the model (in-place) acccording to the new weights
            msg = self.model.image_encoder.model.load_state_dict(theta, strict=False)
            self.logger.info(msg)

            self.save_checkpoint()

            # evaluate the model
            for dataset in self.eval_dataset:
                self.validate(dataset)

    def validate(self, dataset, verbose=True):
        dataset_config = json.load(open(f"conf/datasets/{dataset.name}.json"))
        metric_micro = instantiate(dataset_config["metric"][0]).to(self.device)
        metric_per_class = instantiate(dataset_config["metric"][1]).to(self.device)

        loss = instantiate(dataset_config["loss"]).to(self.device)
        test_loss = 0.0

        self.model.classification_head = get_classification_head(
            self.config, self.model, self.tokenizer, dataset
        )

        self.model.eval()
        with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
            for batch_idx, batch in enumerate(dataset):
                images = batch[0].to(self.device, non_blocking=True)
                targets = batch[1].to(self.device, non_blocking=True)
                output = self.model(images)

                if dataset_config["metric"][0]["task"] == "multiclass":
                    test_loss += loss(output, targets).item()
                    metric_micro.update(torch.argmax(output, dim=1), targets)
                    metric_per_class.update(output, targets)
                else:
                    test_loss += loss(output, targets).item()
                    metric_micro.update(output, targets.long())
                    metric_per_class.update(output, targets.long())

                self.logger.info(
                    f"Test: [{batch_idx * len(images)}/{dataset.num_samples} ({100.0 * batch_idx / dataset.num_batches:.0f}%)]"
                )

        metric_micro = metric_micro.compute()
        metric_per_class = metric_per_class.compute()

        test_loss /= dataset.num_samples
        self.logger.info(
            f"{dataset.name.capitalize()} -> Average loss: {test_loss:.4f}, Micro score: {metric_micro:.4f}"
        )
        if verbose and len(dataset.classnames) < 50:
            newline = "\n"
            self.logger.info(
                "".join(
                    f"{newline} {classname}, {score:.4f}"
                    for classname, score in zip(dataset.classnames, metric_per_class)
                )
            )

    def finalize(self):
        raise NotImplementedError


class AlignmentAgent(BaseAgent):
    def __init__(
        self,
        cfg,
        teacher_model,
        student_model,
        tokenizer,
        task_dataset,
        eval_dataset,
    ):
        super().__init__(cfg)

        self.model = teacher_model
        self.tokenizer = tokenizer
        self.task_dataset = task_dataset
        self.eval_dataset = eval_dataset

        self._init_classification_heads()
        self._init_task_model(teacher_model, student_model)

        self.scaler = GradScaler()
        self.device = self.config["torch"]["device"]
        self.loss = torch.nn.MSELoss().to(self.device)
        self.label_loss = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = instantiate(
            self.config["task"]["optimizer"],
            params=[p for p in self.model.parameters() if p.requires_grad],
        )

        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

    def _init_classification_heads(self):
        for dataset in self.eval_dataset:
            get_classification_head(self.config, self.model, self.tokenizer, dataset)

    def _init_task_model(self, teacher_model, student_model):
        self.model = KDClassifier(
            classification_head=get_classification_head(
                self.config,
                teacher_model,
                self.tokenizer,
                self.task_dataset["train"],
            ),
            teacher_model=ImageEncoder(teacher_model, keep_lang=False),
            student_model=student_model,
        ).to(self.config["torch"]["device"])

        self.model.logit_scale = teacher_model.logit_scale

        self.model.freeze_head()
        self.model.freeze_teacher()
        # self.model.freeze_student()

        self.model = (
            torch.compile(self.model, backend="inductor")
            if self.config["torch"]["torch_compile"]
            else self.model
        )

    def load_checkpoint(self, filename):
        raise NotImplementedError

    def save_checkpoint(self, is_zeroshot=False):
        prefix = "_orig_mod." if self.config["torch"]["torch_compile"] else ""
        suffix = "zs" if is_zeroshot else f"al_{self.current_epoch}"
        state_dict = self.model.state_dict()

        for key in list(state_dict.keys()):
            if f"{prefix}student_model." in key:
                state_dict[key.replace(f"{prefix}student_model.", "")] = state_dict[key]
                del state_dict[key]
            if "teacher_model" in key:
                del state_dict[key]

        ckpt = Path(self.config["env"]["work_dir"]) / "checkpoints"
        ckpt.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
            },
            ckpt
            / f"{self.config['model']['name']}_{self.config['model']['pretrained']}_{suffix}.pt",
        )

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            self.logger.info("SIGTERM received, exiting gracefully...")

    def train(self):
        self.save_checkpoint(is_zeroshot=True)
        for epoch in range(0, self.config["task"]["num_epochs"]):
            self.train_one_epoch()
            self.save_checkpoint()
            if (self.current_epoch + 1) == self.config["task"]["num_epochs"]:
                self.validate(dataset=self.task_dataset["test"])
            elif (epoch + 1) % self.config["task"]["eval_frequency"] == 0:
                self.validate(dataset=self.task_dataset["val"])
            self.current_epoch += 1
        for dataset in self.eval_dataset:
            self.validate_classification(dataset)
        for dataset in self.eval_dataset:
            if dataset.name in ["eurosat_ms", "sen12ms", "bigearthnet19"]:
                self.validate_retieval_mm(dataset)

    def train_one_epoch(self):
        self.model.train()
        scheduler = cosine_lr(
            self.optimizer,
            self.config["task"]["optimizer"]["lr"],
            self.config["task"]["lr_scheduler"]["warmup_length"],
            5 * self.task_dataset["train"].num_batches,
        )

        for batch_idx, (images_rgb, images_ms, targets) in enumerate(
            self.task_dataset["train"]
        ):
            scheduler(
                step=batch_idx
                + self.current_epoch * self.task_dataset["train"].num_batches
            )
            self.optimizer.zero_grad(set_to_none=True)

            images_rgb = images_rgb.to(
                self.config["torch"]["device"], non_blocking=True, dtype=torch.bfloat16
            )
            images_ms = images_ms.to(
                self.config["torch"]["device"], non_blocking=True, dtype=torch.bfloat16
            )
            targets = targets.to(self.config["torch"]["device"], non_blocking=True)

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                teacher_features, student_features, student_outputs = self.model(
                    images_rgb, images_ms
                )
                l1 = self.loss(
                    student_features,
                    teacher_features,
                )

                l2 = self.label_loss(student_outputs, targets.float())
                loss = l1 + l2 * 0.05

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if batch_idx % self.config["torch"]["log_interval"] == 0:
                self.logger.info(
                    f"Train Epoch: {self.current_epoch} [{batch_idx * len(images_rgb)}/{self.task_dataset['train'].num_samples} ({100.0 * batch_idx / self.task_dataset['train'].num_batches:.0f}%)] Loss: {l1.item():.6f} {l2.item():.6f} LR: {self.optimizer.param_groups[0]['lr']}"
                )
            self.current_iteration += 1

    def validate(self):
        pass

    def validate_classification(self, dataset, verbose=True):
        dataset_config = json.load(open(f"conf/datasets/{dataset.name}.json"))
        metric_micro = instantiate(dataset_config["metric"][0]).to(self.device)
        metric_per_class = instantiate(dataset_config["metric"][1]).to(self.device)
        loss = instantiate(dataset_config["loss"]).to(self.device)
        test_loss = 0.0

        eval_model = ImageClassifier(
            image_encoder=self.model.student_model,
            classification_head=get_classification_head(
                self.config, self.model, self.tokenizer, dataset
            ),
        ).to(self.device)

        eval_model.eval()
        with torch.no_grad():
            for batch_idx, (_, images_ms, targets) in enumerate(dataset):
                images_ms = images_ms.to(
                    self.device, non_blocking=True, dtype=torch.bfloat16
                )
                targets = targets.to(self.device, non_blocking=True)
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = eval_model(images_ms)
                    if dataset_config["metric"][0]["task"] == "multiclass":
                        test_loss += loss(output, targets).item()
                        metric_micro.update(torch.argmax(output, dim=1), targets)
                        metric_per_class.update(output, targets)
                    else:
                        test_loss += loss(output, targets).item()
                        metric_micro.update(output, targets.long())
                        metric_per_class.update(output, targets.long())

                self.logger.info(
                    f"Test: [{batch_idx * len(images_ms)}/{dataset.num_samples} ({100.0 * batch_idx / dataset.num_batches:.0f}%)]"
                )

        metric_micro = metric_micro.compute()
        metric_per_class = metric_per_class.compute()

        test_loss /= dataset.num_samples
        self.logger.info(
            f"{dataset.name.capitalize()} -> Average loss: {test_loss:.4f}, Macro AP: {metric_micro:.4f}"
        )
        if verbose and len(dataset.classnames) < 50:
            newline = "\n"
            self.logger.info(
                "".join(
                    f"{newline} {classname}, {score:.4f}"
                    for classname, score in zip(dataset.classnames, metric_per_class)
                )
            )

    def validate_retieval_mm(self, dataset, recall_k_list=[1, 5, 10, 20, 50]):
        def _dataloader_with_indices(dataloader):
            start = 0
            for x, y, z in dataloader:
                end = start + len(x)
                inds = torch.arange(start, end)
                yield x, y, inds
                start = end

        def _recall_at_k(scores, positive_pairs, k):
            nb_texts, nb_images = scores.shape
            topk_indices = torch.topk(scores, k, dim=1)[1]
            nb_positive = positive_pairs.sum(dim=1)
            topk_indices_onehot = torch.nn.functional.one_hot(
                topk_indices, num_classes=nb_images
            )
            positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
            nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(
                dim=(1, 2)
            )
            recall_at_k = nb_true_positive / nb_positive
            return recall_at_k

        def _batchify(func, X, Y, batch_size, device, *args, **kwargs):
            results = []
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                x = X[start:end].to(device)
                y = Y[start:end].to(device)
                result = func(x, y, *args, **kwargs).cpu()
                results.append(result)
            return torch.cat(results)

        eval_model = self.model
        eval_model.eval()

        batch_images_rgb_emb_list = []
        batch_images_ms_emb_list = []
        rgb_ms_index = []
        dataloader = _dataloader_with_indices(dataset)
        for batch_idx, (images_rgb, images_ms, inds) in enumerate(dataloader):
            batch_images_rgb = images_rgb.to(
                self.device, non_blocking=True, dtype=torch.bfloat16
            )
            batch_images_ms = images_ms.to(
                self.device, non_blocking=True, dtype=torch.bfloat16
            )

            batch_ms_rgb_index = [ind for ind, texts in zip(inds, batch_images_ms)]

            with torch.no_grad(), autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_images_rgb_emb, batch_images_ms_emb, _ = eval_model(
                    batch_images_rgb, batch_images_ms
                )
                batch_images_rgb_emb = F.normalize(batch_images_rgb_emb, dim=-1)
                batch_images_ms_emb = F.normalize(batch_images_ms_emb, dim=-1)

            batch_images_rgb_emb_list.append(batch_images_rgb_emb.cpu())
            batch_images_ms_emb_list.append(batch_images_ms_emb.cpu())
            rgb_ms_index.extend(batch_ms_rgb_index)

            self.logger.info(
                f"Test: [{batch_idx * len(batch_images_rgb)}/{dataset.num_samples} ({100.0 * batch_idx / dataset.num_batches:.0f}%)]"
            )

        # batch_size = len(batch_images_emb_list[0])
        batch_size = len(batch_images_rgb_emb_list[0])

        images_rgb_emb = torch.cat(batch_images_rgb_emb_list)
        images_ms_emb = torch.cat(batch_images_ms_emb_list)

        batch_size = 32
        scores = images_ms_emb @ images_rgb_emb.t()

        positive_pairs = torch.zeros_like(scores, dtype=bool)
        positive_pairs[torch.arange(len(scores)), rgb_ms_index] = True
        metrics = {}
        for recall_k in recall_k_list:
            metrics[f"image_rgb_retrieval_recall@{recall_k}"] = (
                (
                    _batchify(
                        _recall_at_k,
                        scores,
                        positive_pairs,
                        batch_size,
                        self.device,
                        k=recall_k,
                    )
                    > 0
                )
                .float()
                .mean()
                .item()
            )
            metrics[f"image_ms_retrieval_recall@{recall_k}"] = (
                (
                    _batchify(
                        _recall_at_k,
                        scores.T,
                        positive_pairs.T,
                        batch_size,
                        self.device,
                        k=recall_k,
                    )
                    > 0
                )
                .float()
                .mean()
                .item()
            )

        self.logger.info(metrics)

    def finalize(self):
        raise NotImplementedError
