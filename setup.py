from setuptools import setup

description = """Johns Hopkins University Cellular Automata Independent Study"""

setup(
    name='JHU-CA',
    version='1.0.0',
    description='A collection of Cellular Automata models completed for an Independent Study course at Johns Hopkins University.',
    long_description=description,
    author='Minh Hua',
    author_email='mhua2@jh.edu',
    license='MIT License',
    keywords='cellular automata',
    url='https://github.com/duyminh1998/',
    packages=[
        'lib',
        'schelling'
    ],
    install_requires=[
        'numpy>=1.10'
    ],
)