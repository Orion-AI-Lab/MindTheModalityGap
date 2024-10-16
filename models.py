import logging

import torch
from utils import pickle_load, pickle_save

logger = logging.getLogger(__name__)


class ImageEncoder(torch.nn.Module):
    def __init__(self, model, keep_lang=False):
        super().__init__()
        self.model = model

        if not keep_lang and hasattr(self.model, "transformer"):
            delattr(self.model, "transformer")

    def forward(self, images):
        assert self.model is not None
        return self.model.encode_image(images)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        logger.info(f"Saving image encoder to {filename}")
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        logger.info(f"Loading image encoder from {filename}")
        return pickle_load(filename)


class ClassificationHead(torch.nn.Linear):
    def __init__(self, normalize, weights, biases=None):
        output_size, input_size = weights.shape
        super().__init__(input_size, output_size)
        self.normalize = normalize
        if weights is not None:
            self.weight = torch.nn.Parameter(weights.clone())
        if biases is not None:
            self.bias = torch.nn.Parameter(biases.clone())
        else:
            self.bias = torch.nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, inputs):
        if self.normalize:
            inputs = inputs / inputs.norm(dim=-1, keepdim=True)
        return super().forward(inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        logger.info(f"Saving classification head to {filename}")
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        logger.info(f"Loading classification head from {filename}")
        return pickle_load(filename)


class ImageClassifier(torch.nn.Module):
    def __init__(self, image_encoder, classification_head):
        super().__init__()
        self.image_encoder = image_encoder
        self.classification_head = classification_head

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def encode_image(self, inputs):
        if hasattr(self.image_encoder, "encode_image"):
            return self.image_encoder.encode_image(inputs)
        else:
            return self.image_encoder(inputs)

    def forward(self, inputs):
        features = self.encode_image(inputs)
        outputs = self.classification_head(features)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def save(self, filename):
        logger.info(f"Saving image classifier to {filename}")
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        logger.info(f"Loading image classifier from {filename}")
        return pickle_load(filename)


class KDClassifier(torch.nn.Module):
    def __init__(self, teacher_model, student_model, classification_head):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.classification_head = classification_head

    def freeze_head(self):
        self.classification_head.weight.requires_grad_(False)
        self.classification_head.bias.requires_grad_(False)

    def freeze_teacher(self):
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

    def freeze_student(self):
        for name, param in self.student_model.named_parameters():
            if "head" not in name:
                param.requires_grad_(False)

    def forward(self, teacher_inputs, student_inputs):
        if self.training:
            self.teacher_model.eval()
            self.classification_head.eval()
            self.student_model.train()

        teacher_features = self.teacher_model(teacher_inputs)
        student_features = self.student_model.encode_image(student_inputs)
        student_outputs = self.classification_head(student_features)
        return teacher_features, student_features, student_outputs

    def __call__(self, teacher_inputs, student_inputs):
        return self.forward(teacher_inputs, student_inputs)

    def save(self, filename):
        logger.info(f"Saving image classifier to {filename}")
        pickle_save(self, filename)

    @classmethod
    def load(cls, filename):
        logger.info(f"Loading image classifier from {filename}")
        return pickle_load(filename)
