import csv
import re

def unpack():
    holder = []
    match = re.compile("^(.+): (.+)")
    with open("vectors.txt", 'r') as infile:
        holding = []
        for line in infile:
            searched = re.search(match, line)
            if searched:
                holding.append(searched[2])
            else:
                holder.append(holding)
                holding = []
    # print(holder)
    return holder

def write2file(holder):
    legend = ["id", "polarity", "subjectivity", "cred_score", "publisher_val", "publisher"]
    with open('vectors.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(legend)
        for each in holder:
            csvwriter.writerow(each)
    
    pass


def main():
    holder = unpack()
    write2file(holder)

if __name__ == '__main__':
    main()