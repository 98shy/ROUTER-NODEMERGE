import sys
import random
from typing import Union, Literal, List
from transformers import AutoTokenizer

class Singleton:
    _instance = None

    @classmethod
    def instance(cls,*args, **kwargs):
        if cls._instance is None:
            cls._instance = cls(*args, **kwargs)
        return cls._instance
    
    def reset(self):
        self.value = 0.0

class Cost(Singleton):
    def __init__(self):
        self.value = 0.0
    def add(self, amount: float):
        try:
            self.value += float(amount)
        except Exception:
            pass

class PromptTokens(Singleton):
    def __init__(self):
        self.value = 0.0
    def add(self, amount: float):
        try:
            self.value += float(amount)
        except Exception:
            pass

class CompletionTokens(Singleton):
    def __init__(self):
        self.value = 0.0
    def add(self, amount: float):
        try:
            self.value += float(amount)
        except Exception:
            pass

class Time(Singleton):
    def __init__(self):
        self.value = ""

class Mode(Singleton):
    def __init__(self):
        self.value = ""

class Tokenizer(Singleton):
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        self.value = ""

class Deepseek_Tokenizer(Singleton):
    def __init__(self, model):
        self.tokenizer = AutoTokenizer.from_pretrained('/path/to/deepseek_v3_tokenizer', use_fast=True)
        self.value = ""