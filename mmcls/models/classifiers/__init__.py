# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseClassifier
from .image import ImageClassifier
from .integrated_classifier import IntegratedClassifier

__all__ = ['BaseClassifier', 'ImageClassifier', 'IntegratedClassifier']
