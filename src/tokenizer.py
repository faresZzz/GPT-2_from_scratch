import re


class SimpleTokenizerV1: 
    def __init__(self, vocab): 

        self.str_to_int = vocab
        self.int_to_str = {v:k for k, v in vocab.items()}

    def encode(self, text): 
        preprocessed_text = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [x for x in preprocessed_text if x.strip()]

        ids = [self.str_to_int[x] for x in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[x] for x in ids])
        # remove spaces before punctuation
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
