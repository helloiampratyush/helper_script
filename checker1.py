import argparse
parser = argparse.ArgumentParser(description="setting hyper parameter")

parser.add_argument('--hidden_unit', type=int, default=10,
                    help='hidden unit in the model')
parser.add_argument('--batch_size', type=int, default=32)
args=parser.parse_args()
HIDDEN_UNIT=args.hidden_unit
BATCH_SIZE=args.batch_size
print(HIDDEN_UNIT,BATCH_SIZE)