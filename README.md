<p align="center">
  <img src="regioselect/static/image/logo.png"/>
</p>

---

RegioSelect is a web application for the prediction of C-H activation.

[TRY RegioSelect here: http://www.regioselect.org](http://www.regioselect.org)

## Installation

For the installation, we recommend using `conda` to get all the necessary dependencies:

    conda env create -f etc/environment_regioselect.yml && conda activate regioselect


Then download the binaries of xTB version 6.5.1:

    cd regioselect/scripts/dep/; tar -xvf ./xtb-6.5.1-linux-x86_64.tar.xz; cd ../../..
