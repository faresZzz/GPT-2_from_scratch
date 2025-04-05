import torch 
from torch.utils.data import Dataset


class GPTDatasetPretrainingOverText(Dataset):
    """
    Dataset class for pretraining GPT-style models on raw text.
    This class processes a text corpus into overlapping chunks of tokens
    for autoregressive language modeling. It creates input-target pairs
    where the target is the input shifted by one token position to the right,
    enabling the model to learn to predict the next token in a sequence.
    Parameters
    ----------
    txt : str
        The raw text corpus to process.
    tokenizer : Tokenizer
        A tokenizer instance that provides an `encode` method to convert text to token IDs.
        Should handle the "<|endoftext|>" special token if present.
    max_length : int
        Maximum sequence length for each training example.
    stride : int
        Number of tokens to stride when creating overlapping sequences.
        Smaller stride values result in more overlap between training examples.
    Attributes
    ----------
    input_ids : list of torch.tensor
        List of token ID sequences used as inputs.
    target_ids : list of torch.tensor
        List of token ID sequences used as targets (shifted by one position).
    Returns
    -------
    When indexed, returns a tuple of (input_ids, target_ids) tensors for the requested example.
    """

    def __init__(self, txt, tokenizer, max_length, stride, allowed_special={"<|endoftext|>"}):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special=allowed_special)

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
class SpamDataset(Dataset):
    """
    A PyTorch Dataset for spam classification tasks.
    This class prepares text data for spam classification by tokenizing the texts,
    handling padding and truncation to ensure uniform sequence lengths, and
    converting labels to tensors.
    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing at least two columns:
        - 'Text': The text content to be classified
        - 'Label': The class labels (0 for non-spam, 1 for spam)
    tokenizer : object
        Tokenizer object with an 'encode' method that converts text to token IDs
    max_length : int, optional
        Maximum sequence length for the encoded texts.
        If None, the length of the longest text in the dataset is used.
        If specified, longer sequences will be truncated.
    pad_token_id : int, default=50256
        Token ID to use for padding shorter sequences
    Methods
    -------
    __getitem__(index) : tuple
        Returns a tuple of (encoded_text_tensor, label_tensor) for the given index
    __len__() : int
        Returns the total number of samples in the dataset
    _longest_encoded_length() : int
        Helper method to find the length of the longest encoded sequence
    Examples
    --------
    >>> tokenizer = YourTokenizer()
    >>> df = pd.DataFrame({'Text': ['spam email', 'normal email'], 'Label': [1, 0]})
    >>> dataset = SpamDataset(df, tokenizer)
    >>> encoded_text, label = dataset[0]
    """

    def __init__(self, dataframe, tokenizer, max_length=None, pad_token_id=50256):
        self.data = dataframe

        # Pre-tokenize texts
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
            ]

    def __getitem__(self, index):
        """
        Fetches a single sample from the dataset at the given index.
        This method implements the PyTorch Dataset interface, allowing the dataset 
        to be indexed with dataset[i].
        Args:
            index (int): The index of the sample to retrieve.
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: The encoded text as a long tensor
                - torch.Tensor: The corresponding label as a long tensor
        """

        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        """
        Find the length of the longest encoded text in the dataset.
        This method iterates through all encoded texts stored in the instance
        and determines the maximum length among them.
        Returns:
            int: The length of the longest encoded text sequence.
        Note:
            This is an internal helper method as indicated by the leading underscore.
        """

        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            self.encoded_texts.append(
                tokenizer.encode(self._format_alpaca(entry))
            )

    def _format_alpaca(self, entry): 
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
        response_text = f"\n\n### Response:\n{entry['output']}"
        full_text = instruction_text + input_text + response_text
        return full_text

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)