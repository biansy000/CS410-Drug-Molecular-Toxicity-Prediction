import argparse

def parse_config(argv=None):
    parser = argparse.ArgumentParser(description='Model parameters')

    parser.add_argument('--sum', default=0)

    args = parser.parse_args()
    return args

