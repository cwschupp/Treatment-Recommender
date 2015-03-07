import re
import sys

fname = sys.argv[1]

f = open(fname)
f_new = open('new_diagnosis.tsv', 'w')

for line in f:
    new_line = re.sub('[^0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c]', '', line)
    f_new.write(new_line)
