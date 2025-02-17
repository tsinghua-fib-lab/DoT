
## Initial Setup

1. Clone the repository:
```bash
git clone https://github.com/princeton-nlp/WebShop
```

2. Follow the setup instructions from the original WebShop README.md to configure the dataset and environment. Use the entire products list for the experiment.

3. Put all python files in the `WebShop` directory.

## Configuration Changes Required

Before running the experiments, make the following modifications at the beginning of each file start with `webshop_`:

1. Update the LLaMA client URL:
   - Locate the LLaMA client configuration
   - Replace the default URL with your server URL

2. Modify the WebShop URL:
   - Default: `localhost:3000/ABC`
   - For the line with `WEBSHOP_URL = "Change to your customized URL"`, replace the URL with your WebShop Environment URL

3. Update Openai API key:
   - Replace the key with your Openai API key in the setOpenAi function in `utils.py`
  
# Dataset Building

For test set generation, run the following command:
```bash
python webshop_testset_building.py
```
- This will output a test set of 100 tasks, which recorded the steps for each task to model allocation.
- The output file is `webshop_succeed_finetune_data_testset.json`.

For training set generation, run the following command:
```bash
python webshop_trainset_building.py
```
- The current training is set to run first 10 tasks, each 20 times, and the successful purchase will be recorded.
  - For more training data:
    - Edit `n` in `run_episodes(prompt, n)` to change the number of tasks.
    - Edit `max_attempts` in `adaptive_threshold_run()` to change the number of attempts for each task.

# Formatting Data for Training

To format the data for training, first run `webshop_trainset_building.py` to generate the training data. Then, use the `formatting_building.ipynb` notebook to format the data for training.

# Formatting Data for Testing

1. After the model reallocation, run the second part of the `formatting_building.ipynb` notebook to format the data for testing. This is the reallocation model selections with task substeps for each task.
2. For files start with `webshop_dot`, use the formatted data for testing.

Notes:
   - `updated_first_file.json` is the model allocation data for our experiment.

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
