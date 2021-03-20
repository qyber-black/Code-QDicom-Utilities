# QDicom Utilities

A set of utilities to deal with dicom files and data repositories, specifically
aimed at the qyber archives, but some may be more generally useful or adaptable.

* read_dicom_siemens.py
  * Display/write DICOM info/image/spectroscopy, specifically for Siemens IMA files.
  * Read dicom functions for import in other tools.
  * This can specifically deal with Siemens CSA headers.
* check-dataset-structure.py
  * Check and possibly fix file system structure of a dicom dataset.

See the --help messages and the comments in the files for more information.

## Dependencies

The scripts are tested with python3.8 and later. You need the following packages:

* pydicom
* numpy
* matplotlib
* xxhash

(see requirements.txt, installable via `pip3 install -r requirements.txt`).

## Locations

The code is developed and maintained on [qyber\\black](https://qyber.black)
at https://qyber.black/pca/code-qdicom-utilities

This code is mirrored at
* https://github.com/xis10z/Code-QDicom-Utilities

The mirrors are only for convenience, accessibility and backup.

## People

* [Frank C Langbein](https://qyber.black/xis10z), [School of Computer Science and Informatics](https://www.cardiff.ac.uk/computer-science), [Cardiff University](https://www.cardiff.ac.uk/); [langbein.org](https://langbein.org/)

## License

Copyright (C) 2021, Cardiff University  
Author: Frank C Langbein <frank@langbein.org>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
