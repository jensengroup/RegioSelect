#!/bin/bash

wget https://sourceforge.net/projects/jmol/files/Jmol/Version%2016.1/Jmol%2016.1.41/Jmol-16.1.41-binary.zip

unzip Jmol-16.1.41-binary.zip

cd jmol-16.1.41
unzip jsmol.zip

cd ..

mv jmol-16.1.41/jsmol jsmol

rm -r jmol-16.1.41
rm Jmol-16.1.41-binary.zip