import re

#matches one or more non-word characters (anything other than a-z,A-Z,0-9 and _)
# keep words and remove symbols and numbers
WORD_REGEX = re.compile(r'(\w+)')
#SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')



#sentence as input, tokenize into words
def tokenize(sentence):
    # Convert to lowercase and split the sentence into tokens.
    tokens = WORD_REGEX.findall(sentence.lower())
    # removing strip whitespace
    #tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens


#loading a vocabulary file where each line has a single word
#def load_str_list(fname):
    with open(fname) as f:
    #creating an empty list
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines


def load_str_list(fname):
    with open(fname, 'r') as f:
        return [line.strip() for line in f]
  
class VocabDict:

    def __init__(self, vocab_file):
        self.word_list = load_str_list(vocab_file)
        self.word2idx_dict = {w:n_w for n_w, w in enumerate(self.word_list)}
        self.vocab_size = len(self.word_list)
        #self.unk2idx = self.word2idx_dict['<unk>'] if '<unk>' in self.word2idx_dict else None
        self.unk2idx = self.word2idx_dict.get('<unk>', None)
        
    def idx2word(self, n_w):
        return self.word_list[n_w]
    
    
    #def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.unk2idx is not None:
            return self.unk2idx
        else:
            raise ValueError('word %s not in dictionary (while dictionary does not contain <unk>)' % w)

    def word2idx(self, w):
        return self.word2idx_dict.get(w, self.unk2idx)

    def tokenize_and_index(self, sentence):
        # This assumes that the tokenize function is defined outside the class
        inds = [self.word2idx(w) for w in tokenize(sentence)]
    
        return inds
    
    
    
#get the reference from the  https://github.com/tbmoon/basic_vqa/blob/master/utils/text_helper.py

