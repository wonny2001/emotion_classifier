import random
import sys
import argparse
import csv
import string
import time


OUT_FILE='out/test0422.csv'


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
        #clock 0
        elif column == 'anger0':
            generators.append(lambda: random.uniform(0.009886557,0.20480545))
        elif column == 'disgust0':
            generators.append(lambda: random.uniform(0.00007, 0.014853356))
        elif column == 'happy0':
            generators.append(lambda: random.uniform(0.06398171, 0.6509571))
        elif column == 'surprise0':
            generators.append(lambda: random.uniform(0.018697215, 0.390521))
        elif column == 'sad0':
            generators.append(lambda: random.uniform(0.014285376, 0.17149168))
        elif column == 'neutral0':
            generators.append(lambda: random.uniform(0.22317481, 0.82267475))
        elif column == 'result0':
            generators.append(lambda: '0') #random.choice([1, 0])

        #gallery 1
        elif column == 'anger1':
            generators.append(lambda: random.uniform(0.029739598, 0.59183836))
        elif column == 'disgust1':
            generators.append(lambda: random.uniform(0.003188006, 0.051654585))
        elif column == 'happy1':
            generators.append(lambda: random.uniform(0.042635635, 0.2546624))
        elif column == 'surprise1':
            generators.append(lambda: random.uniform(0.014592985, 0.10824544 ))
        elif column == 'sad1':
            generators.append(lambda: random.uniform(0.01849962, 0.19864453))
        elif column == 'neutral1':
            generators.append(lambda: random.uniform(0.23245348, 0.67217124))
        elif column == 'result1':
            generators.append(lambda: '1') #random.choice([1, 0])

        #sns 2
        elif column == 'anger2':
            generators.append(lambda: random.uniform(0.003180297, 0.18568122))
        elif column == 'disgust2':
            generators.append(lambda: random.uniform(0.000240074, 0.00642167))
        elif column == 'happy2':
            generators.append(lambda: random.uniform(0.11639011, 0.9253144))
        elif column == 'surprise2':
            generators.append(lambda: random.uniform(0.009221391, 0.49379063))
        elif column == 'sad2':
            generators.append(lambda: random.uniform(0.001832818, 0.061653424))
        elif column == 'neutral2':
            generators.append(lambda: random.uniform(0.044488143, 0.3353962))
        elif column == 'result2':
            generators.append(lambda: '2') #random.choice([1, 0])

        #game 3
        elif column == 'anger3':
            generators.append(lambda: random.uniform(0.003903895, 0.20120052))
        elif column == 'disgust3':
            generators.append(lambda: random.uniform(0.000008, 0.02355388))
        elif column == 'happy3':
            generators.append(lambda: random.uniform(0.013327818, 0.29578075))
        elif column == 'surprise3':
            generators.append(lambda: random.uniform(0.011825203, 0.42018622))
        elif column == 'sad3':
            generators.append(lambda: random.uniform(0.022996562, 0.8208493))
        elif column == 'neutral3':
            generators.append(lambda: random.uniform(0.1424127, 0.8091867))
        elif column == 'result3':
            generators.append(lambda: '3') #random.choice([1, 0])

        #official 4
        elif column == 'anger4':
            generators.append(lambda: random.uniform(0.008649088, 0.55435634))
        elif column == 'disgust4':
            generators.append(lambda: random.uniform(0.020066181, 0.26661786))
        elif column == 'happy4':
            generators.append(lambda: random.uniform(00.039123125, 0.33754954))
        elif column == 'surprise4':
            generators.append(lambda: random.uniform(0.003154741, 0.030702827))
        elif column == 'sad4':
            generators.append(lambda: random.uniform(00.00619683, 0.120209165))
        elif column == 'neutral4':
            generators.append(lambda: random.uniform(00.05310515, 0.8890979))

        elif column == 'result4':
            generators.append(lambda: '4') #random.choice([1, 0])

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
    parser.add_argument('rows', type=int, default='100',
                        help='number of rows to generate')
    parser.add_argument('--delimiter', type=str, default=',', required=False,
                        help='the CSV delimiter')
    parser.add_argument('schema', type=str, nargs='+', default='float',
                        # choices=['int', 'str', 'float'],
                        help='list of column types to generate')

    args = parser.parse_args()
    integer_csv(args.rows, args.schema, args.delimiter)