import pathlib
import sentencepiece as spm

from tokenwiser.tok._tok import Tok


class SentencePieceTokenizer(Tok):
    def __init__(self, model_file):
        mod = spm.SentencePieceProcessor(model_file=model_file)
        self.model = SentencePieceTokenizer(vocab_size=mod.vocab_size(), model_type="", mod=mod)

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
