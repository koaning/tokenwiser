import pathlib

import sentencepiece as spm


class SentencePieceTokenizer:
    def __init__(self, vocab_size=1000, model_type="bpe", mod_name=None, prefix="", mod=None, output_type=str):
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.mod_name = mod_name
        self.prefix = prefix
        self.mod = mod
        self.output_type = output_type

    @classmethod
    def from_file(cls, model_file):
        mod = spm.SentencePieceProcessor(model_file=model_file)
        return SentencePieceTokenizer(vocab_size=mod.vocab_size(), model_type="", mod=mod)

    def transform(self, X, y=None):
        return [self.mod.encode(x, out_type=self.output_type) for x in X]

    def fit(self, X, y=None):
        if not self.mod:
            mod_name = f"{self.prefix}-{self.model_type}-{self.vocab_size}"
            spm.SentencePieceTrainer.train(sentence_iterator=X.__iter__(),
                                           model_prefix=mod_name,
                                           vocab_size=self.vocab_size,
                                           model_type=self.model_type)
            self.mod = spm.SentencePieceProcessor(model_file=mod_name + ".model")
        return self

    @classmethod
    def train_file(cls, input_file, vocab_size=10_000, model_type="bpe", mod_name=None):
        """
        model_type= "word", "bpe", "unigram", "char"
        """
        if not mod_name:
            mod_name = f"{pathlib.Path(input_file).stem}-{model_type}-{vocab_size}"
        spm.SentencePieceTrainer.train(input=input_file,
                                       model_prefix=mod_name,
                                       vocab_size=vocab_size,
                                       model_type=model_type)
        return mod_name + ".model"
