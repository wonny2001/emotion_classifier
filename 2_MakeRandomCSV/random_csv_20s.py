import random
import sys
import argparse
import csv
import string
import time


OUT_FILE='out/test0611_20s.csv'


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
            generators.append(lambda: random.uniform(0.009886557,0.10480545))
        elif column == 'disgust0':
            generators.append(lambda: random.uniform(0.00027, 0.012853356))
        elif column == 'happy0':
            generators.append(lambda: random.uniform(0.04395171, 0.5509571))
        elif column == 'surprise0':
            generators.append(lambda: random.uniform(0.028697215, 0.590521))
        elif column == 'sad0':
            generators.append(lambda: random.uniform(0.024285376, 0.225168))
        elif column == 'neutral0':
            generators.append(lambda: random.uniform(0.32317481, 0.92267475))
        elif column == 'result0':
            generators.append(lambda: '0') #random.choice([1, 0])

        #gallery 1
        elif column == 'anger1':
            generators.append(lambda: random.uniform(0.049739598, 0.49183836))
        elif column == 'disgust1':
            generators.append(lambda: random.uniform(0.002188006, 0.031654585))
        elif column == 'happy1':
            generators.append(lambda: random.uniform(0.052635635, 0.2746624))
        elif column == 'surprise1':
            generators.append(lambda: random.uniform(0.114592985, 0.20824544 ))
        elif column == 'sad1':
            generators.append(lambda: random.uniform(0.11849962, 0.29864453))
        elif column == 'neutral1':
            generators.append(lambda: random.uniform(0.33245348, 0.87217124))
        elif column == 'result1':
            generators.append(lambda: '1') #random.choice([1, 0])

        #sns 2
        elif column == 'anger2':
            generators.append(lambda: random.uniform(0.023180297, 0.19568122))
        elif column == 'disgust2':
            generators.append(lambda: random.uniform(0.001240074, 0.00742167))
        elif column == 'happy2':
            generators.append(lambda: random.uniform(0.01639011, 0.8253144))
        elif column == 'surprise2':
            generators.append(lambda: random.uniform(0.019221391, 0.48379063))
        elif column == 'sad2':
            generators.append(lambda: random.uniform(0.011832818, 0.041653424))
        elif column == 'neutral2':
            generators.append(lambda: random.uniform(0.034488143, 0.3453962))
        elif column == 'result2':
            generators.append(lambda: '2') #random.choice([1, 0])

        #game 3
        elif column == 'anger3':
            generators.append(lambda: random.uniform(0.004903895, 0.20140052))
        elif column == 'disgust3':
            generators.append(lambda: random.uniform(0.010008, 0.03355388))
        elif column == 'happy3':
            generators.append(lambda: random.uniform(0.033327818, 0.31578075))
        elif column == 'surprise3':
            generators.append(lambda: random.uniform(0.021825203, 0.412018622))
        elif column == 'sad3':
            generators.append(lambda: random.uniform(0.032996562, 0.8508493))
        elif column == 'neutral3':
            generators.append(lambda: random.uniform(0.1454127, 0.806867))
        elif column == 'result3':
            generators.append(lambda: '3') #random.choice([1, 0])

        #official 4
        elif column == 'anger4':
            generators.append(lambda: random.uniform(0.078649088, 0.59435634))
        elif column == 'disgust4':
            generators.append(lambda: random.uniform(0.120066181, 0.26661786))
        elif column == 'happy4':
            generators.append(lambda: random.uniform(0.139123125, 0.43754954))
        elif column == 'surprise4':
            generators.append(lambda: random.uniform(0.103154741, 0.130702827))
        elif column == 'sad4':
            generators.append(lambda: random.uniform(0.05619683, 0.150209165))
        elif column == 'neutral4':
            generators.append(lambda: random.uniform(0.03310515, 0.8590979))

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