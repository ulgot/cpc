#!/bin/basho

cat cpu.nfo > index.html
for f in *png; do
  echo "<p><img src=$f></p>" >> index.html
done

tidy -m index.html
