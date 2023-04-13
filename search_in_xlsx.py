#!/usr/bin/python

import os, sys
import openpyxl


def find(path, word):
    l = []
    d = os.listdir(path)
    for file in d:
        filename = str(path) + "/" + str(file)
        print("Finding %s in %s" % (word, file))
        if filename.endswith(".xlsx"):
            wb = openpyxl.load_workbook(filename=filename)
            ws = wb.active
            for row in ws.rows:
                for cell in row:
                    if cell.value and str(word) in str(cell.value):
                        l.append((file, cell))
    if l:
        print("Word %s found %d times in:" % (word, len(l)))
        for fn, cell in l:
            print("File: %s, row: %s ,column: %s" % (fn, cell.row, cell.column))
    else:
        print("Word %s not found" % word)


if __name__ == "__main__":
    try:
        find(sys.argv[1], sys.argv[2])
    except IndexError:
        print("\tExecute: python searchpy <path> <word>")
        print("\tEg: python searchpy /home/user/files/ Fox")
