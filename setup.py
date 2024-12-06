from setuptools import setup

setup(
    name='automatic prompt engineering',
    version='0.1',
    description='',
    author='Jihua',
    author_email='liougehooa@163.com',
    packages=['automatic_prompt_engineer',
              'automatic_prompt_engineer.evaluation'],
    package_data={'automatic_prompt_engineer': ['configs/*']},
    install_requires=[
        'numpy',
        'openai',
        'fire',
        'tqdm',
        'asyncio',
        'nest-asyncio',
        'tenacity',
        'gradio'
    ],
    tests_require=[
        'pytest',
    ],
)
