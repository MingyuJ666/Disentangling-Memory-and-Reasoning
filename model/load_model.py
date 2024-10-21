import torch
import os
from typing import Dict, Optional, Sequence, Union, Callable

from transformers import AutoModelForCausalLM, PreTrainedModel
import torch.nn.functional as F
from model.sparse_models import SparseGPT2LMHeadModel, SparseLlamaForCausalLM


class InputEmbedding(torch.nn.Module):
    def __init__(self, original_embedding, n_new_tokens, initialize_tokens=None):
        super(InputEmbedding, self).__init__()
        self.original_embedding = original_embedding
        self.num_original_tokens = original_embedding.weight.size(0)
        print("original vocab size: ", self.num_original_tokens)
        self.n_new_tokens = n_new_tokens
        if n_new_tokens > 0:
            self.new_embedding = torch.nn.Embedding(n_new_tokens, 
                original_embedding.weight.size(1)).to(original_embedding.weight.device)
            if initialize_tokens is not None:
                new_embeddings = self.original_embedding(initialize_tokens)
                self.new_embedding.weight.data = new_embeddings
            else:
                self.new_embedding.weight.data = original_embedding.weight.mean(
                    dim=0, keepdim=True).repeat(n_new_tokens, 1)
        else:
            self.new_embedding = None

    def forward(self, input_ids):
        if input_ids.max() >= self.num_original_tokens:
            if input_ids.min() >= self.num_original_tokens:
                return self.new_embedding(input_ids - self.num_original_tokens)
            else:
                prompt_mask = input_ids >= self.num_original_tokens
                text_mask = input_ids < self.num_original_tokens
                prompt_embd = self.new_embedding(input_ids[prompt_mask] - self.num_original_tokens)
                original_embd = self.original_embedding(input_ids[text_mask])
                all_embd = torch.zeros((input_ids.size(0), input_ids.size(1), 
                        self.original_embedding.weight.size(1)), 
                        dtype=original_embd.dtype,
                        device=input_ids.device)
                all_embd[prompt_mask.unsqueeze(-1).repeat(1, 1,
                        self.original_embedding.weight.size(1))] = prompt_embd.flatten()
                all_embd[text_mask.unsqueeze(-1).repeat(1, 1, 
                        self.original_embedding.weight.size(1))] = original_embd.flatten()
                return all_embd
        else:
            return self.original_embedding(input_ids)
        

class OutputEmbedding(torch.nn.Module):
    def __init__(self, original_linear, n_new_tokens, initialize_tokens=None):
        super(OutputEmbedding, self).__init__()
        self.original_linear = original_linear
        self.n_new_tokens = n_new_tokens
        if n_new_tokens > 0:
            self.new_linear = torch.nn.Linear(original_linear.weight.size(1), 
                n_new_tokens).to(original_linear.weight.device)
            if initialize_tokens is not None:
                new_embeddings = F.embedding(initialize_tokens, 
                                             self.original_linear.weight.data)
                self.new_linear.weight.data = new_embeddings
            else:
                self.new_linear.weight.data = original_linear.weight.mean(dim=0, 
                    keepdim=True).repeat(n_new_tokens, 1)
        else:
            self.new_linear = None

    def forward(self, inputs):
        original_token_logits = self.original_linear(inputs)
        if self.n_new_tokens > 0:
            new_token_logits = self.new_linear(inputs)
            return torch.cat((original_token_logits, new_token_logits), dim=-1)
        else:
            return original_token_logits
        

def load_embeddings(model, input_embedding_file, output_embedding_file, 
                    n_tokens, orig_vocab_size):
    assert os.path.isfile(input_embedding_file)
    new_token_embeddings = torch.load(input_embedding_file)
    print(new_token_embeddings)
    try:
        if new_token_embeddings.weight.size(0) == n_tokens + orig_vocab_size:
            model.set_input_embeddings(new_token_embeddings)
        elif new_token_embeddings.weight.size(0) == n_tokens:
            model.set_input_embeddings(InputEmbedding(
                model.get_input_embeddings(), n_tokens))
            model.get_input_embeddings().new_embedding = \
                new_token_embeddings
        else:
            print("new token embeddings size does not match: ", 
                    new_token_embeddings.weight.size(0))
            exit(1)
    except:
        assert new_token_embeddings.size(0) == n_tokens + orig_vocab_size
        model.get_input_embeddings().weight.data = new_token_embeddings
    print("input embeddings loaded from file")

    if output_embedding_file is not None:
        assert os.path.isfile(output_embedding_file)
        new_token_embeddings = torch.load(output_embedding_file)
        if new_token_embeddings.weight.size(0) == n_tokens + orig_vocab_size:
            model.set_output_embeddings(new_token_embeddings)
        elif new_token_embeddings.weight.size(0) == n_tokens:
            model.set_output_embeddings(OutputEmbedding(
                model.get_output_embeddings(), n_tokens))
            model.get_output_embeddings().new_linear = \
                new_token_embeddings
        else:
            print("new token embeddings size does not match: ", 
                    new_token_embeddings.weight.size(0))
            exit(1)
        print("output embeddings loaded from file")
    else:
        model.tie_weights()
    
            
def save_pretrained(
    self,
    save_directory: Union[str, os.PathLike],
    **kwargs,
):
    if os.path.isfile(save_directory):
        raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")
    os.makedirs(save_directory, exist_ok=True)
    torch.save(self.get_input_embeddings().new_embedding, 
                os.path.join(save_directory, "input_embeddings.pt"))
    torch.save(self.get_output_embeddings().new_linear, 
                os.path.join(save_directory, "output_embeddings.pt"))

class MyAutoModelForCausalLM(AutoModelForCausalLM):

    def __init__(self, n_tokens=0, sparse=False, 
                 parameter_efficient_mode=False, **kwargs):
        self = super().__init__(**kwargs)
        self.n_tokens = n_tokens
        self.sparse = sparse
        self.parameter_efficient_mode = parameter_efficient_mode
        
    @classmethod
    def from_pretrained(cls, n_tokens=0, input_embedding_file=None, output_embedding_file=None,
                        sparse=False, parameter_efficient_mode='none',
                        prompt_tokens=None, initialize_tokens=None, **kwargs):
        # monkey patching
        if parameter_efficient_mode == 'prompt-tuning':
            PreTrainedModel.save_pretrained = save_pretrained

        if sparse:
            if 'llama' in kwargs['pretrained_model_name_or_path'] or 'alpaca' in kwargs['pretrained_model_name_or_path']:
                model = SparseLlamaForCausalLM.from_pretrained(**kwargs)

            elif 'gpt2' in kwargs['pretrained_model_name_or_path']:
                model = SparseGPT2LMHeadModel.from_pretrained(**kwargs)
            else:
                raise NotImplementedError
        else:
            model = AutoModelForCausalLM.from_pretrained(**kwargs, trust_remote_code=True)

        model.n_tokens = n_tokens
        model.sparse = sparse
        model.parameter_efficient_mode = parameter_efficient_mode
        model.prompt_tokens = prompt_tokens

        if n_tokens > 0:

            orig_vocab_size = model.get_input_embeddings().weight.size(0)
            print("original vocab size: ", orig_vocab_size)

            if initialize_tokens is not None:
                initialize_tokens = torch.tensor(initialize_tokens, 
                                    dtype=torch.long, device=model.device)
    
            if parameter_efficient_mode != 'none':

                model.config.vocab_size = orig_vocab_size + n_tokens

                if input_embedding_file is not None:
                    load_embeddings(model, input_embedding_file, output_embedding_file, 
                                    n_tokens, orig_vocab_size)
                else:

                    model.set_input_embeddings(InputEmbedding(
                        model.get_input_embeddings(), n_tokens, initialize_tokens))
                    model.set_output_embeddings(OutputEmbedding(
                        model.get_output_embeddings(), n_tokens, initialize_tokens))
                    
            elif initialize_tokens is not None:

                model.resize_token_embeddings(orig_vocab_size + n_tokens)
                new_vocab_size = model.get_input_embeddings().weight.size(0)
                assert new_vocab_size == n_tokens + orig_vocab_size

                new_embeddings = model.get_input_embeddings()(initialize_tokens)
                model.get_input_embeddings().weight.data[-n_tokens:] = new_embeddings

                new_embeddings = F.embedding(initialize_tokens, 
                                             model.get_output_embeddings().weight.data)
                model.get_output_embeddings().weight.data[-n_tokens:] = new_embeddings

        return model
    

if __name__ == "__main__":
    model = MyAutoModelForCausalLM.from_pretrained(n_tokens=10,
        pretrained_model_name_or_path="../pretrained_models/llama-7b-hf",
        device_map="auto", load_in_8bit=True,
        offload_folder="offload", offload_state_dict = True)
    
    print(model)