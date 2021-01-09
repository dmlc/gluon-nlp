import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='Merge the results into one csv')
    parser.add_argument('')
    return parser

def main(args):
    pass

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
