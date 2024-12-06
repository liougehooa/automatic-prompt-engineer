import pytest
from automatic_prompt_engineer.template import DemosTemplate

def test_fill_demos_single_demo():
    demo_template = "[INPUT] -> [OUTPUT]"
    data = (["input1"], ["output1"])
    expected_result = "input1 -> output1"
    
    dt = DemosTemplate(demo_template)
    result = dt.fill_demos(data)
    
    assert result == expected_result

def test_fill_demos_multiple_demos():
    demo_template = "[INPUT] -> [OUTPUT]"
    data = (["input1", "input2"], ["output1", "output2"])
    expected_result = "input1 -> output1\n\ninput2 -> output2"
    
    dt = DemosTemplate(demo_template)
    result = dt.fill_demos(data)
    
    assert result == expected_result

def test_fill_demos_custom_delimiter():
    demo_template = "[INPUT] -> [OUTPUT]"
    data = (["input1", "input2"], ["output1", "output2"])
    expected_result = "input1 -> output1--input2 -> output2"
    
    dt = DemosTemplate(demo_template, delimiter="--")
    result = dt.fill_demos(data)
    
    assert result == expected_result

def test_fill_demos_empty_data():
    demo_template = "[INPUT] -> [OUTPUT]"
    data = ([], [])
    expected_result = ""
    
    dt = DemosTemplate(demo_template)
    result = dt.fill_demos(data)
    
    assert result == expected_result