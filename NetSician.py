import click
import threading
import time
import src.Operator as operator
from src.GenerationNetwork import *

def lel():
    print("lel")

def main():
    operator.load_and_train_lead()

if __name__ == "__main__":
    main()