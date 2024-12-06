import pytest
from unittest.mock import MagicMock, patch

from automatic_prompt_engineer.evaluation import elo
from automatic_prompt_engineer import llm

def test_expected_score():
    r1 = 1500
    r2 = 1600
    expected = 1 / (1 + 10 ** ((r2 - r1) / 400))
    result = elo.ELOEvaluator._expected_score(r1, r2)
    assert result == expected

def test_update_elo_win():
    r1 = 100
    r2 = 100
    score = 1  # Player 1 wins
    k = 32
    e1 = elo.ELOEvaluator._expected_score(r1, r2)
    e2 = elo.ELOEvaluator._expected_score(r2, r1)
    expected_r1 = r1 + k * (score - e1)
    expected_r2 = r2 + k * ((1 - score) - e2)
    updated_r1, updated_r2 = elo.ELOEvaluator._update_elo(r1, r2, score, k)
    assert updated_r1 == expected_r1
    assert updated_r2 == expected_r2
    assert updated_r1 > updated_r2
    
def test_update_elo_lose():
    r1 = 100
    r2 = 100
    score = 0  # Player 2 wins
    k = 32
    e1 = elo.ELOEvaluator._expected_score(r1, r2)
    e2 = elo.ELOEvaluator._expected_score(r2, r1)
    expected_r1 = r1 + k * (score - e1)
    expected_r2 = r2 + k * ((1 - score) - e2)
    updated_r1, updated_r2 = elo.ELOEvaluator._update_elo(r1, r2, score, k)
    
    assert updated_r1 == expected_r1
    assert updated_r2 == expected_r2
    assert updated_r1 < updated_r2

def test_update_elo_draw():
    r1 = 100
    r2 = 100
    score = 0.5  # Draw
    k = 32
    e1 = elo.ELOEvaluator._expected_score(r1, r2)
    e2 = elo.ELOEvaluator._expected_score(r2, r1)
    expected_r1 = r1 + k * (score - e1)
    expected_r2 = r2 + k * ((1 - score) - e2)
    updated_r1, updated_r2 = elo.ELOEvaluator._update_elo(r1, r2, score, k)
    assert updated_r1 == expected_r1
    assert updated_r2 == expected_r2
    assert updated_r1 == updated_r2

def test_update_elo_not_draw_from_history():
    r1 = 200 # initial not draw
    r2 = 100 # initial not draw
    score = 0.5  # Draw
    k = 32
    e1 = elo.ELOEvaluator._expected_score(r1, r2)
    e2 = elo.ELOEvaluator._expected_score(r2, r1)
    expected_r1 = r1 + k * (score - e1)
    expected_r2 = r2 + k * ((1 - score) - e2)
    updated_r1, updated_r2 = elo.ELOEvaluator._update_elo(r1, r2, score, k)
    assert updated_r1 == expected_r1
    assert updated_r2 == expected_r2
    assert updated_r1 > updated_r2

    
def test_elo_evaluation_result_sorted():
    prompts_ratings = {'prompt1': 1200, 'prompt2': 1300, 'prompt3': 1100}
    result = elo.EloEvaluationResult('test_task', prompts_ratings)
    expected_sorted = [('prompt2', 1300), ('prompt1', 1200), ('prompt3', 1100)]
    assert result.sorted() == expected_sorted

def test_elo_evaluation_result_top_n():
    prompts_ratings = {'prompt1': 1200, 'prompt2': 1300, 'prompt3': 1100}
    result = elo.EloEvaluationResult('test_task', prompts_ratings)
    expected_top_n = [('prompt2', 1300), ('prompt1', 1200)]
    assert result.top_n(2) == expected_top_n
    

def test_generate_prompts_output():
    # Mock evaluation configuration
    eval_config = {
        'elo_k': 32,
        'gen_model': {'gpt_config':{}},
        'score_model': {'gpt_config':{}},
        'ranking_system_prompt': '',
        'user_template': '',
        'batch_size': 3,
        'num_samples': 2
    }
    
    # Mock llm.model_from_config
    with patch('automatic_prompt_engineer.llm.model_from_config') as mock_model_from_config:
        # Create mock models to return
        mock_gen_model = MagicMock()
        mock_score_model = MagicMock()
        
        
        # Set the return values for model_from_config
        mock_model_from_config.side_effect = [mock_gen_model, mock_score_model]
        
        # Create an instance of ELOEvaluator
        evaluator = elo.ELOEvaluator(eval_config)
        
    # Mock llm.model_from_config
    with patch('automatic_prompt_engineer.template.EvalTemplate') as eval_template_mock:  
        gen_count = 0  
        score_count = 0 
        
        def gen_side_effect():
            global gen_count
            gen_count += 1
            return [
                {"role": "system", "content": 'Generate prompt'},
                {"role": "user", "content": f'Input {gen_count}'}
                ]
        
        def score_side_effect():
            global score_count
            score_count += 1
            return [
                {"role": "system", "content": 'Score prompt'},
                {"role": "user", "content": f'Input {score_count}'}
                ]
            
        # Create mock models to return
        eval_template = eval_template_mock.return_value
        eval_template.fill_generate_messages = MagicMock(side_effect= gen_side_effect)
        eval_template.fill_score_messages = MagicMock(side_effect= score_side_effect)
   
    # Mock the llm_gen_model
    evaluator.llm_gen_model = MagicMock()
    evaluator.llm_gen_model.complete_prompt.return_value = [
        MagicMock(message=MagicMock(content='Generated prompt 1')),
        MagicMock(message=MagicMock(content='Generated prompt 2')),
    ]

    # Mock the llm_score_model
    evaluator.llm_score_model = MagicMock()
    evaluator.llm_score_model.complete_prompt.side_effect = [
        [MagicMock(message=MagicMock(content='A'))],
        [MagicMock(message=MagicMock(content='B'))],
        [MagicMock(message=MagicMock(content='A'))],
        [MagicMock(message=MagicMock(content='B'))],
    ]

    # Define test prompts and inputs
    prompts = ['prompt1', 'prompt2']
    inputs = ['input1', 'input2']

    # Call the generate_prompts_output method
    all_scores = evaluator.generate_prompts_output('test_task', prompts, inputs)

    # Assertions to verify the outputs
    assert isinstance(all_scores, list)
    assert len(all_scores) == 2  # 2 combinations * 2 inputs
    for score in all_scores:
        assert 'input' in score
        assert 'prompt1' in score
        assert 'prompt2' in score
        assert 'avg_score' in score
        
        
def test_rating_scores():
    # Mock configuration for ELOEvaluator
    eval_config = {
        'elo_k': 32,
        'gen_model': {},
        'score_model': {},
        'ranking_system_prompt': '',
        'user_template': '',
    }

    # Mock llm.model_from_config
    with patch('automatic_prompt_engineer.llm.model_from_config') as mock_model_from_config:
        # Create mock models to return
        mock_gen_model = MagicMock()
        mock_score_model = MagicMock()
        
        
        # Set the return values for model_from_config
        mock_model_from_config.side_effect = [mock_gen_model, mock_score_model]
        
        # Create an instance of ELOEvaluator
        evaluator = elo.ELOEvaluator(eval_config)
        
        # Define test prompts
    prompts = ['prompt1', 'prompt2', 'prompt3']

    # Define test all_scores
    # Each entry represents the comparison between two prompts
    # 'score1' and 'score2' are the scores assigned by the evaluator
    all_scores = [
        {'prompt1': 'prompt1', 'prompt2': 'prompt2', 'avg_score': 1},   # prompt1 beats prompt2
        
        {'prompt1': 'prompt1', 'prompt2': 'prompt3', 'avg_score': 0.5}, # prompt1 draws with prompt3
        
        {'prompt1': 'prompt2', 'prompt2': 'prompt3', 'avg_score': 0},   # prompt2 loses to prompt3
    ]

    # Call the rating_scores method
    prompt_ratings = evaluator.rating_scores(prompts, all_scores)

    # Check that all prompts have ratings
    assert 'prompt1' in prompt_ratings
    assert 'prompt2' in prompt_ratings
    assert 'prompt3' in prompt_ratings

    # Verify that the ratings have been updated correctly
    # Since exact ratings depend on the ELO formula, we perform relative comparisons
    assert prompt_ratings['prompt1'] > prompt_ratings['prompt2'], "Prompt1 should have a higher rating than Prompt2"
    assert prompt_ratings['prompt3'] > prompt_ratings['prompt2'], "Prompt3 should have a higher rating than Prompt2"
    
    # Prompt1 and Prompt3 drew, so their ratings should be close
    assert abs(prompt_ratings['prompt1'] - prompt_ratings['prompt3']) < 1, "Prompt1 and Prompt3 ratings should be close"