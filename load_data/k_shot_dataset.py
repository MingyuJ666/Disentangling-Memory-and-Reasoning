import random, copy, os, json
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
from torch.utils.data import Dataset
import transformers
import torch
import numpy as np
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import sentence_transformers

class KshotDataset(Dataset):
    """Dataset for in-context learning."""

    def __init__(self, dataset: Dataset, demo_dataset: Dataset, k=4, 
                 demo_selection='uniform', 
                 selection_model=None, tokenizer=None, prompt_text=None, save_dir=None):
        super(KshotDataset, self).__init__()
        self.data = dataset
        self.demo_data = demo_dataset
        self.demo_embeddings = None
        self.k = k
        self.selection_model = selection_model
        self.tokenizer = tokenizer
        self.prompt_text = prompt_text
        self.save_dir = save_dir
        self.sorted_demo_data = None
        if demo_selection == 'uniform':
            self.selection_func = self.uniform_selection
        elif demo_selection == 'prompt':
            self.selection_func = self.prompt_selection
        elif demo_selection == 'similar':
            self.selection_func = self.similarity_selection
        else:
            raise NotImplementedError
        self.x = []
        self.y = []
        
        for i in range(len(self.data)):
            demo_data = self.selection_func(i)
            demos_text = '\n\n'.join([d['x'] + d['y'] for d in demo_data])
            input_text = demos_text + '\n\n' + self.data[i]['x']
            output_text = self.data[i]['y']
            self.x.append(input_text)
            self.y.append(output_text)
        
    def is_correct(self, model_completion, gt_example):
        return self.data.is_correct(model_completion, gt_example)

    def uniform_selection(self, index=None):
        rand_ids = random.sample(range(len(self.demo_data)), self.k)
        return [self.demo_data[i] for i in rand_ids]
    
    def similarity_selection(self, index):
        self.embedding_model = sentence_transformers.SentenceTransformer('all-mpnet-base-v2')
        self.embedding_model.eval()
        if self.demo_embeddings is None:
            demo_embedding_file = 'embeddings/train.npy'
            if os.path.isfile(demo_embedding_file):
                self.demo_embeddings = np.load(demo_embedding_file)
            else:
                os.makedirs('embeddings', exist_ok=True)
                with torch.no_grad():
                    self.demo_embeddings = self.embedding_model.encode(
                        [d["x"] + d["y"] for d in self.demo_data])
                np.save(demo_embedding_file, self.demo_embeddings)

        with torch.no_grad():
            test_embeddings = self.embedding_model.encode([self.data[index]["x"] + self.data[index]["y"]])

        sims = sentence_transformers.util.cos_sim(test_embeddings, self.demo_embeddings)
        sims = sims[0].cpu().detach().numpy()

        del self.embedding_model

        return [self.demo_data[i] for i in np.argsort(sims)[-self.k:][::-1]]
        
    def prompt_selection(self, index=None):
        sorted_demo_file = self.save_dir + '/sorted_demo_data.json'
        if self.sorted_demo_data is None:
            if os.path.exists(sorted_demo_file):
                with open(sorted_demo_file, 'r') as f:
                    self.sorted_demo_data = json.load(f)
            else:
                self.selection_model.eval()
                loss_fct = CrossEntropyLoss()
                labels = torch.tensor(self.tokenizer(self.prompt_text)['input_ids']).to('cuda')
                all_loss = []
                for d in self.demo_data:
                    inputs = [d['x'] + d['y'] + self.prompt_text]
                    encoding = self.tokenizer(inputs, padding=True, return_tensors='pt').to('cuda')
                    with torch.no_grad():
                        logits = self.selection_model(**encoding).logits
                        prompt_logits = logits[0, -len(labels):]
                        loss = loss_fct(prompt_logits.view(-1, prompt_logits.size(-1)), labels.view(-1)) 
                        all_loss.append(loss.sum().item())
                os.makedirs(self.save_dir, exist_ok=True)
                np.save(self.save_dir + '/loss.npy', np.array(all_loss))
                print(all_loss)
                self.sorted_demo_data = [self.demo_data[i] for i in np.argsort(all_loss)]
                with open(sorted_demo_file,'w') as f:
                    json.dump(self.sorted_demo_data, f)
        return self.sorted_demo_data[:self.k]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(x=self.x[i], y=self.y[i])