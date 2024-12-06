import random
from automatic_prompt_engineer import generate, evaluate, config, template, data, llm
from automatic_prompt_engineer.evaluation import elo, bandits


def evaluator_from_config(configuaration):
    """Returns a model based on the config."""
    method = configuaration["method"]
    if method == "bandits":
        return bandits.BanditsEvaluator(configuaration)
    
    if method == "elo":
        return elo.ELOEvaluator(configuaration)
    else:
        raise ValueError(f"Noe implement eval method: {method}")
    
def generator_from_config(configuaration):
    """Returns a model based on the config."""
    return generate.SimpleLLMGenerator(configuaration)


def simple_ape(task, dataset, config_file='configs/bandits.yaml'):
    """
    Function that wraps the find_prompts function to make it easier to use.
    Design goals: include default values for most parameters, and automatically
    fill out the config dict for the user in a way that fits almost all use cases.

    The main shortcuts this function takes are:
    - Uses the same dataset for prompt generation, evaluation, and few shot demos
    - Uses UCB algorithm for evaluation
    - Fixes the number of samples per prompt per round to 5
    Parameters:
        dataset: The dataset to use for evaluation.
        config_file: load ape config from file.
    Returns:
        An evaluation result and a function to evaluate the prompts with new inputs.
    """
    conf = config.simple_config(config_file)
    return find_prompts(task, prompt_gen_data=dataset, eval_data=dataset, conf=conf)

def find_prompts(task,
                 prompt_gen_data,
                 eval_data,
                 conf):
    """
    Function to generate prompts using APE.
    Parameters:
        prompt_gen_data: The data to use for prompt generation.
        eval_data: The data to use for evaluation.
        conf: The configuration dictionary.
    Returns:
        An evaluation result. Also returns a function to evaluate the prompts with new inputs.
    """
    
    generator = generator_from_config(conf['generation'])
    print('Generating prompts...')
    prompts = generator.generate_prompts(task, prompt_gen_data)

    print(f"Model returned {len(prompts)} generated prompts.")
    print('Deduplicating...')
    prompts = list(set(prompts))
    print('Deduplicated to {} prompts.'.format(len(prompts)))

    print(f'Evaluating generated {len(prompts)} prompts...')

    ## eval template
    # [prompt] is where the prompt will be inserted.
    # [full_DEMO] is where the full demo will be inserted.
    # [INPUT] is where the input to the first demo will be inserted.
    # [OUTPUT] is where the output from the first demo will be inserted.
    
    inputs, _ = eval_data
    evaluator = evaluator_from_config(conf['evaluation'])
    res = evaluator.rate_prompts(task=task,
                                 prompts=prompts,
                                 inputs=inputs)

    return res