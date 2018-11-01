import spacy

class BasicTokenizer(object):
    """
    Normalizes text for WordPieceTokenizer
    """
    def __init__(self, lower_case=True):
        self.lower_case = lower_case
        self.tokenizer = spacy.blank("en")
    
    def tokenize(self, sequence):
        """
        Tokenizes (and lowercases) input sequence with SpaCy.
        """
        if self.lower_case:
            tokens =  [t.text.lower() for t in self.tokenizer(sequence)]
        else:
            tokens = [t.text for t in self.tokenizer(sequence)]
        return " ".join(tokens)


class WordPieceTokenizer(object):
    def __init__(self, vocab, unk_token="[UNK]", max_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_chars_per_word = max_chars_per_word

    def tokenize(self, sequence):
        output_tokens = []
        for s in sequence.split():
            if len(list(s)) > self.max_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
            return output_tokens
