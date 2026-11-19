import io

import sentencepiece as spm
import torch

from alethic.model.transformer import ModelConfig, Transformer
from alethic.sft import chat

TINY = dict(name="tiny", attention="gqa", n_layers=2, d_model=32, n_heads=4, n_kv_heads=2, head_dim=8,
            ffn_hidden=64, vocab_size=300, max_seq_len=64)


def test_reply_streams_text_stops_at_limit_and_returns_what_it_printed(tmp_path):
    (tmp_path / "c.txt").write_text("\n".join("hello world %d <user> </user> <assistant> </assistant>" % i for i in range(300)))
    spm.SentencePieceTrainer.train(input=str(tmp_path / "c.txt"), model_prefix=str(tmp_path / "t"), vocab_size=300,
                                   model_type="unigram", byte_fallback=True, bos_id=1, eos_id=2, unk_id=0, pad_id=3,
                                   normalization_rule_name="identity", add_dummy_prefix=False)
    sp = spm.SentencePieceProcessor(model_file=str(tmp_path / "t.model"))
    model = Transformer(ModelConfig.from_dict(TINY)).eval()
    out = io.StringIO()
    text = chat.reply(model, sp, "<user>\nhello\n</user>\n", "cpu", max_new=5, out=out)
    assert out.getvalue().rstrip("\n") == text
    assert len(sp.encode(text)) <= 6
