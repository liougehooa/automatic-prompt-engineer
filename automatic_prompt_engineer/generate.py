from tqdm import tqdm

import asyncio
import nest_asyncio

from automatic_prompt_engineer import data, llm, template

nest_asyncio.apply()


class SimpleLLMGenerator:
    """Prompt generator with LLM."""
    
    def __init__(self, gen_config):
        self.gen_config = gen_config
        self.llm_model = llm.model_from_config(gen_config['model'])
        self.gen_template = template.GenerationTemplate(gen_config['system_prompt'], 
                                                        gen_config['user_template'], 
                                                        gen_config['demo_template'])
        
    def generate_prompts(self, task, gen_data):
        """
        Generates prompts using the prompt generator.
        Parameters:
            task: The task to generate prompt.
            gen_data: The data to use for prompt generation.
        Returns:
            A list of prompts with size n * num_subsamples.
        """
        
        ## queries are a list of few shot prompt whose size is equal to num_subsamples,
        ## each sample/prompt is a filled with few shots of num_demos
        
        messages_of_prompts = []
        for _ in range(self.gen_config['num_gen_prompts']):
            subsampled_data = data.subsample_data(gen_data, self.gen_config['num_samples_per_gen_prompt'])
            
            ## each query is a prompt with few shot demos
            messages = self.gen_template.fill(task, subsampled_data)
            messages_of_prompts.append(messages)
        
        # generate text prompts. each prompt with num_prompts_per_subsample choices
        generated_prompts = self.gen_prompts(messages_of_prompts)
        return generated_prompts

    def gen_prompts(self, messages_of_prompts):
        """
        Generate prompts for the prompt generator.
        """
        batch_size = self.gen_config['batch_size']
        
        print(f'generated by batch size: {batch_size}')
        ## slice by batch size
        sampled_messages_batches = [messages_of_prompts[i:i + batch_size]
                                        for i in range(0, len(messages_of_prompts), batch_size)]
        
        generated_prompts = []
        for messages_batch in tqdm(sampled_messages_batches, desc='Generating prompts'):
            prompts = []
            asyncio.run(self.batch_generation(messages_batch, prompts))
            generated_prompts.extend(prompts)
                    
        return generated_prompts

    async def generate_prompt(self, messages, prompts):
        choices = self.llm_model.complete_prompt(messages=messages,
                                            config=self.gen_config['model']['gpt_config'])
        
        prompts.extend([choice.message.content for choice in choices])
            
    async def batch_generation(self, messages_batch, prompts):
        """
        Generate prompts for the prompt generator.
        """
        await asyncio.gather(*[self.generate_prompt(messages, prompts) for messages in messages_batch])