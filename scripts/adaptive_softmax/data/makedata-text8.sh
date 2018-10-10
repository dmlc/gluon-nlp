wget http://mattmahoney.net/dc/text8.zip -O text8.gz
gzip -d text8.gz -f

head text8 -c 99000000 > train
tail text8 -c 1000000 > valid

tr " " '\n' < train | sort | uniq -c | awk '{if ($1>10) print $2;}' | tr "\n" " " > voc

cat voc > trainx
echo $'' >> trainx
cat train >> trainx
awk '{if (NR == 1) {for (a=1;a<=NF;a++) cn[$a]++; b=0;} else for (a=1;a<=NF;a++) {b++; if ((b%1000) == 0) print ""; if (cn[$a]) printf $a " "; else printf "<UNK> ";}}' < trainx > train2
mv train2 ./text8.train.txt
rm trainx

cat voc > validx
echo $'' >> validx
cat valid >> validx
awk '{if (NR == 1) {for (a=1;a<=NF;a++) cn[$a]++; b=0;} else for (a=1;a<=NF;a++) {b++; if ((b%1000) == 0) print ""; if (cn[$a]) printf $a " "; else printf "<UNK> ";}}' < validx > valid2
mv valid2 ./text8.test.txt
rm validx

rm voc
rm train
rm valid
rm text8
