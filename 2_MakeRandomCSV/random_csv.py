import random
import sys
import argparse
import csv
import string
import time


OUT_FILE='out/clock.csv'

def integer_csv(rows, schema, delimiter):
    random.seed(time.time())
    generators = []
    char_set = (string.ascii_letters + string.digits +
                '"' + "'" + "#&* \t")

    for column in schema:
        if column == 'int':
            generators.append(lambda: random.randint(0, 1e9))
        elif column == 'str':
            generators.append(lambda: ''.join(
                random.choice(char_set) for _ in range(12)))
        elif column == 'float':
            generators.append(lambda: random.random())
        elif column == 'anger':
            generators.append(lambda: random.uniform(0.03,0.27))
        elif column == 'disgust':
            generators.append(lambda: random.uniform(0.05,0.25))
        elif column == 'happy':
            generators.append(lambda: random.uniform(0.04,0.53))
        elif column == 'surprise':
            generators.append(lambda: random.uniform(0.001,0.05))
        elif column == 'sad':
            generators.append(lambda: random.uniform(0.01,0.4))
        elif column == 'neutrial':
            generators.append(lambda: random.uniform(0.26,0.5))
        elif column == 'result':
            generators.append(lambda: '0') #random.choice([1, 0])

    # writer = csv.writer(sys.stdout, delimiter=delimiter)
    #writer = csv.writer(open("validate.csv","wb"))
    writer = csv.writer(open(OUT_FILE,"a"))
    for x in xrange(rows):
        writer.writerow([g() for g in generators])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a large CSV file.',
        epilog='''"Space is big. You just won't believe how vastly,
        hugely, mind-bogglingly big it is."''')
    parser.add_argument('rows', type=int,
                        help='number of rows to generate')
    parser.add_argument('--delimiter', type=str, default=',', required=False,
                        help='the CSV delimiter')
    parser.add_argument('schema', type=str, nargs='+',
                        # choices=['int', 'str', 'float'],
                        help='list of column types to generate')

    args = parser.parse_args()
    integer_csv(args.rows, args.schema, args.delimiter)