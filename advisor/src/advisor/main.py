#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from advisor.crew import Advisor

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'user_goal': '1.5M AUD total portfolio value in 5 years',
        'timeline': '5 years',
        'risk': 'moderate',
        'current_date': str(datetime.now().date()),
        'sector':'Tech'
    }
    
    try:
        Advisor().crew().kickoff(inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'user_goal': '1.5M AUD total portfolio value in 5 years',
        'timeline': '5 years',
        'risk': 'moderate',
        'current_date': str(datetime.now().date()),
        'sector':'Tech'
    }
    try:
        Advisor().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Advisor().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        'user_goal': '1.5M AUD total portfolio value in 5 years',
        'timeline': '5 years',
        'risk': 'moderate',
        'current_date': str(datetime.now().date()),
        'sector':'Tech'
    }
    
    try:
        Advisor().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")
