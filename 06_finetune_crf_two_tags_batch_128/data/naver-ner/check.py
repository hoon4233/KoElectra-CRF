TAGS = ["ORG-B", "ORG-I","LOC-B", "LOC-I"]

ORI_FILE_PATH = 'test.tsv'

with open(ORI_FILE_PATH,'r') as f :
    LINES = f.readlines()

NEW_LINES = []
count = 0

for idx, line in enumerate(LINES) :
    sentence, tags = line.split('\t')

    tags = tags.split()

    for idx, tag in enumerate(tags) :
        if tag in TAGS :
            count += 1

print(count)