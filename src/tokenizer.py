"""
Simple tokenizer for maze action sequences.
Vocabulary: <pad>, <s>, </s>, <unk>, R, U
"""


class SimpleTokenizer:
    """
    Simple tokenizer for maze sequences.
    
    Tokens:
        <pad>: Padding (ID=0)
        <s>: Beginning of sequence (BOS, ID=1)
        </s>: End of sequence (EOS, ID=2)
        <unk>: Unknown token (ID=3)
        R: Move right (ID=4)
        U: Move up (ID=5)
    """
    
    def __init__(self):
        self.vocab = {
            '<pad>': 0,
            '<s>': 1,
            '</s>': 2,
            '<unk>': 3,
            'R': 4,
            'U': 5
        }
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
    
    def __len__(self):
        return len(self.vocab)
    
    def token_to_id(self, token):
        """Convert token to ID"""
        return self.vocab.get(token, self.unk_token_id)
    
    def id_to_token_str(self, token_id):
        """Convert ID to token string"""
        return self.id_to_token.get(token_id, '<unk>')
    
    def encode(self, sequence):
        """
        Encode a sequence of tokens to IDs.
        
        Args:
            sequence: List of token strings, e.g., ['R', 'R', 'U']
        
        Returns:
            List of token IDs with BOS and EOS added
        """
        tokens = ['<s>'] + sequence + ['</s>']
        return [self.token_to_id(t) for t in tokens]
    
    def decode(self, token_ids):
        """
        Decode token IDs back to token strings.
        
        Args:
            token_ids: List or tensor of token IDs
        
        Returns:
            List of token strings
        """
        if isinstance(token_ids, list):
            return [self.id_to_token_str(tid) for tid in token_ids]
        else:
            # Handle tensor
            return [self.id_to_token_str(tid.item()) for tid in token_ids]
    
    def decode_to_string(self, token_ids):
        """
        Decode token IDs to a single string.
        
        Args:
            token_ids: List or tensor of token IDs
        
        Returns:
            String of tokens separated by spaces
        """
        tokens = self.decode(token_ids)
        return ' '.join(tokens)
