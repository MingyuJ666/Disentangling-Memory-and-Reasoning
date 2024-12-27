## Preprocess

### Preprocess the dataset into reason and Memory pattern
1️⃣ First: Replace the GPT4's API key in data_agent.py and data_agent_order.py.

If you want to generate the memory step and the reason step naturally without order:
```
python data_agent.py --mode "train" --dataset "StrategyQA"
```
If you want first to generate the memory step and then generate the reason step:
```
python data_agent_order.py --mode "train" --dataset "StrategyQA"
```
2️⃣ However, I have uploaded all the datasets we ran in the dataset_folder, which you should take and use.

3️⃣ You can choose the different datasets and choose a train or test pattern.

### Check the dataset in preprocess.py
Class StrategyQAData_Ours, TruthfulQAData_Ours, and CommonsenseQAData_Ours are the class we need.
