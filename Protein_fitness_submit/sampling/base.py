from abc import ABC, abstractmethod
import numpy as np
import torch

class Algo(ABC):
    '''
    net: Discrete Diffusion Model
    forward_op: Forward Operator
    data_config: Data Configuration
    n_max_mutations: Maximum number of mutations for the generations
    device: Device to run the model
    '''
    def __init__(self, net, forward_op=None, data_config=None, n_max_mutations=None, device='cuda', causal_LM=False):
        self.net = net
        self.forward_op = forward_op
        self.data_config = data_config
        self.n_max_mutations = n_max_mutations
        self.device = device

        self.alphabet = list(data_config.alphabet[:20]) #only take the 20 standard amino acids
        self.special_tokens = list(data_config.alphabet[20:])

        self.seq_len = data_config.seq_len
        self.full_seq = [self.net.tokenizer.tokenize(s) for s in data_config.full_seq] if not causal_LM else data_config.full_seq
        self.full_seq_string = data_config.full_seq

        if data_config.residues is None:
            self.residues = list(range(self.seq_len))
        else:
            self.residues = [i-1 for i in data_config.residues] #convert to 0-indexing
        self.all_residues = list(range(self.seq_len))

        self.n_residues = len(self.residues)
        self.mask = torch.zeros(self.seq_len)
        self.mask[self.residues] = 1
        self.mask = self.mask.to(device).int()
        if not causal_LM:
            self.full_seq = torch.from_numpy(np.array(self.full_seq)).to(device).int().squeeze(-1)

    def project_sequences(self, sequences):
        #remove special characters (assume this is rare)
        #alternatively, could just toss out sequences that have bad characters
        new_sequences = []
        for seq in sequences:
                for s in seq:
                    if s not in self.alphabet:
                        seq = seq.replace(s, np.random.choice(self.alphabet))
                new_sequences.append(seq)

        if self.n_residues > 4 and self.n_max_mutations is not None:
            for i, seq in enumerate(new_sequences):
                #choose n_max_mutations residues and limit mutations to those positions
                mutated_residues = [j for j, (r1, r2) in enumerate(zip(seq, self.full_seq_string)) if r1 != r2]
                max_mutations = min(self.n_max_mutations, len(mutated_residues))
                mutated_residues = np.random.choice(mutated_residues, max_mutations, replace=False)
                #mutated_residues = np.random.choice(self.residues, self.n_max_mutations, replace=False)
                seq = list(seq)
                for j, r in enumerate(seq):
                    if j not in mutated_residues:
                        seq[j] = self.full_seq_string[j]
                new_sequences[i] = ''.join(seq)

        return new_sequences

    @abstractmethod
    def inference(self, observation=None, num_samples=1, **kwargs):
        '''
        Args:
            - observation: observation for one single ground truth
            - num_samples: number of samples to generate for each observation
        '''
        pass