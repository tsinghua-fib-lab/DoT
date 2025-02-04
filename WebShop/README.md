
## Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/princeton-nlp/WebShop
```

2. Follow the setup instructions from the original WebShop README.md to configure the dataset and environment.

3. Put all python files in the `WebShop` directory.

## Configuration Changes Required

Before running the experiments, make the following modifications:

1. Update the LLaMA client URL:
   - Locate the LLaMA client configuration
   - Replace the default URL with your server URL

2. Modify the WebShop URL:
   - Default: `localhost:3000/ABC`
   - Change to your customized URL

3. Update Openai API key:
   - Replace the key with your Openai API key in the setOpenAi functio in `utils.py`
  
# Dataset Building

For test set generation, run the following command:
```bash
python webshop_testset_building.py
```

For training set generation, run the following command:
```bash
python webshop_trainset_building.py
```

Note: After building the dataset, use the formatting_building jupyter notebook to format the dataset.

## Running Different Approaches

### COT (Chain of Thought)
```bash
python webshop_cot.py
```

### TOT (Tree of Thoughts)
```bash
python webshop_tot.py
```

### DataShunt
```bash
python webshop_datashunt.py
```

### DOT (Division of Thoughts)
```bash
python webshop_dot.py
```
