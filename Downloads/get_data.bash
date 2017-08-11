# Run from inside the Downloads folder.

preprocess_exec=./tokenizer.sed

SNLI='https://nlp.stanford.edu/projects/snli/snli_1.0.zip'
glovepath='http://nlp.stanford.edu/data/glove.840B.300d.zip'
infersentPath='https://s3.amazonaws.com/senteval/infersent/infersent.allnli.pickle'

# GloVe
echo $glovepath
curl -LO $glovepath
7za x glove.840B.300d.zip 
rm glove.840B.300d.zip

#infersent pickle
echo $infersentPath
curl -Lo infersent.allnli.pickle $infersentPath

### download SNLI
echo $SNLI
mkdir SNLI/true
curl -o SNLI/snli_1.0.zip $SNLI
unzip SNLI/snli_1.0.zip -d SNLI

for split in train dev test
do
    fpath=SNLI/$split.snli.txt
    awk '{ if ( $1 != "-" ) { print $0; } }' SNLI/snli_1.0/snli_1.0_$split.txt | cut -f 1,6,7 | sed '1d' > $fpath
    cut -f1 $fpath > SNLI/true/labels.$split
    cut -f2 $fpath | $preprocess_exec > SNLI/true/s1.$split
    cut -f3 $fpath | $preprocess_exec > SNLI/true/s2.$split
    rm $fpath
done
rm SNLI/snli_1.0.zip
rm -r SNLI/__MACOSX
rm -r SNLI/snli_1.0

