# QDicom Utilities

> SPDX-FileCopyrightText: Copyright (C) 2020-2023 Frank C Langbein <frank@langbein.org>, Cardiff University  
> SPDX-License-Identifier: AGPL-3.0-or-later  

A set of utilities to deal with dicom files and data repositories, specifically
aimed at the qyber archives, but some may be more generally useful or adaptable.

* read_dicom_siemens.py
  * Display/write DICOM info/image/spectroscopy, specifically for Siemens IMA files.
  * Read dicom functions for import in other tools.
  * This can specifically deal with Siemens CSA headers.
* check-dataset-structure.py
  * Check and possibly fix file system structure of a dicom dataset.
* create_dataset.py
  * Creates dataset from dicom repository
* conv_picai.py
  * Creates dataset from PI-CAI repositories

See the --help messages and the comments in the files for more information.

See `requirements.txt` for dependencies. It's installable via
`pip3 install -r requirements.txt`.

## Locations

The code is developed and maintained on [qyber\\black](https://qyber.black)
at https://qyber.black/pca/code-qdicom-utilities

This code is mirrored at
* https://github.com/qyber-black/Code-QDicom-Utilities

The mirrors are only for convenience, accessibility and backup.

## People

* [Frank C Langbein](https://qyber.black/xis10z), [School of Computer Science and Informatics](https://www.cardiff.ac.uk/computer-science), [Cardiff University](https://www.cardiff.ac.uk/); [langbein.org](https://langbein.org/)
