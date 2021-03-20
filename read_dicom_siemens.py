#!/usr/bin/env python3
#
# read_dicom_siemens.py
# Display/write DICOM info/image/spectroscopy, specifically for Siemens IMA files.
# Read dicom functions for import.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Copyright (C) 2020-2021, Cardiff University
# Author: Frank C Langbein <frank@langbein.org>

import os
import sys
import struct
import json
import argparse
import re

import pydicom
import pydicom.uid
import numpy
import matplotlib.pyplot as plt

NUMPY_DATA_TYPE = numpy.complex64

# DICOM tags
TAG_SOP_CLASS_UID = (0x0008, 0x0016)
# Siemens tags
TAG_CONTENT_TYPE         = (0x0029, 0x1008)
TAG_SPECTROSCOPY_DATA    = (0x7fe1, 0x1010)
TAG_SPECTROSCOPY_DATA_VB = (0x5600, 0x0020)

def main():
  p = argparse.ArgumentParser(description='Display/write DICOM info/image, specifically for Siemens IMA files')
  p.add_argument('-c','--cmap', action='store_true', help='display color dicom image (or write to _cmap.png file if -w is given, overwrites files)')
  p.add_argument('-g','--gray', action='store_true', help='display gray dicom image (or write to _gray.png file if -w is given, overwrites files)')
  p.add_argument('-o','--overlay', action='store_true', help='display dicom overlays')
  p.add_argument('-s','--spectrum', action='store_true', help='show spectrum')
  p.add_argument('-w','--write', action='store_true', help='write dicom info into .asc file (overwrites existing files)')
  p.add_argument('dicom', nargs='+', help='IMA file name or directory (recursively) to process')
  args = p.parse_args()
  err = False
  for d in args.dicom:
    if not (os.path.isfile(d) or os.path.isdir(d)):
      sys.stderr.write(os.path.basename(__file__)+': '+d+' is not a file or directory\n')
      err = True
  if not err:
    for d in args.dicom:
      if os.path.isdir(d):
          process_dir(d,args.write,args.cmap,args.gray,args.overlay,args.spectrum)
      else:
        display_dicom(d,args.write,args.cmap,args.gray,args.overlay,args.spectrum)

def process_dir(dir,write,cmap,gray,overaly,spectrum):
  # Scan directory for dicoms and process them
  for d in sorted(os.listdir(dir)):
    p = os.path.join(dir,d)
    if os.path.isdir(p):
      process_dir(p,write,cmap,gray,overaly,spectrum)
    elif d.lower().endswith('.ima'):
      display_dicom(p,write,cmap,gray,overaly,spectrum)

def pretty_dict(d, indent=0):
  # Pretty print dict
  for key, value in d.items():
    print('\t' * indent + str(key))
    if isinstance(value, dict):
      pretty_dict(value, indent+1)
    else:
      print('\t' * (indent+1) + str(value) + ":" + str(type(value)))

def display_dicom(filename,write,cmap,gray,overlay,spectrum):
  # Display/save dicom information
  print("Processing %s" % filename)
  dicom, result = read_dicom(filename)
  if write:
    out = open(os.path.splitext(filename)[0]+'.json',"w")
  else:
    out = sys.stdout
  print(json.dumps(result, indent=4, sort_keys=True), file=out)
  if write:
    out.close()
  if overlay:
    overlay_img = {}
    for idx in range(0,0x1f):
      try:
        overlay_img[idx] = dicom.overlay_array(0x6000+idx)
      except:
        pass
  if cmap:
    # Plot using perceptually uniform sequential colormaps (viridis, plasma, inferno, magma or cividis)
    try:
      plt.imshow(dicom.pixel_array, cmap=plt.cm.viridis)
      if overlay:
        for k in overlay_img:
          plt.imshow(overlay_img[k], alpha=.5, cmap=plt.cm.inferno)
      if write:
          plt.savefig(os.path.splitext(filename)[0]+'_cmap.png')
      else:
          plt.show()
    except:
      print("No pixel data")
  if gray:
    # Plot using grayscale
    try:
      plt.imshow(dicom.pixel_array, cmap='gray')
      if overlay:
        for k in overlay_img:
          plt.imshow(overlay_img[k], alpha=.5, cmap=plt.cm.inferno)
      if write:
        plt.savefig(os.path.splitext(filename)[0]+'_gray.png')
      else:
        plt.show()
    except:
      print("No pixel data")
  if not gray and not cmap and overlay:
    try:
      for k in overlay_img:
        plt.imshow(overlay_img[k], cmap=plt.cm.inferno)
      if write:
        plt.savefig(os.path.splitext(filename)[0]+'_overlay.png')
      else:
        plt.show()
    except:
      print("No overlay data")
  if spectrum:
    try:
      data = dicom[TAG_SPECTROSCOPY_DATA].value
    except:
      print("No spectrum")
      data = None
    if data is not None:
      data = struct.unpack("<%df" % (len(data) / 4), data)
      ax = plt.subplot(2,2,(1,2))
      ax.plot([abs(complex(data[i], data[i+1])) for i in range(0, len(data), 2)])
      ax = plt.subplot(2,2,3)
      ax.plot([data[i] for i in range(0, len(data), 2)])
      ax = plt.subplot(2,2,4)
      ax.plot([data[i] for i in range(1, len(data), 2)])
      if write:
        plt.savefig(os.path.splitext(filename)[0]+'_spectrum.png')
      else:
        plt.show()

def dicom_elements(ds):
  # Extract non-binary dicom elements; filtered
  dont = ['PixelData', 'Overlay Data', 'DataSetTrailingPadding',
          '(7fe1, 0010)', '[CSA Data]',
          '[MedCom OOG Info]', '[MedCom History Information]' ]
  duplicate_keys = []
  contents = {}
  for de in ds:
    if de.keyword != "" and de.keyword != "[Unknown]":
      key = de.keyword
    elif de.name != "[Unknown]" and de.name != "Private Creator":
      key = de.name
    else:
      key = str(de.tag)
    value = None
    if de.tag in [(0x0029, 0x1010), (0x0029, 0x1110), (0x0029, 0x1210)]:
      # Siemens Image Header Info
      # (0x0029, 0x__10) is one of several possibilities
      # - SIEMENS CSA NON-IMAGE, CSA Data Info
      # - SIEMENS CSA HEADER, CSA Image Header Info
      # - SIEMENS CSA ENVELOPE, syngo Report Data
      # - SIEMENS MEDCOM HEADER, MedCom Header Info
      # - SIEMENS MEDCOM OOG, MedCom OOG Info (MEDCOM Object Oriented Graphics)
      # Pydicom identifies it as "CSA Image Header Info"
      try:
        value = parse_csa_header(get_tag(ds, de.tag))
      except ValueError:
        pass
    elif de.tag in [(0x0029, 0x1020), (0x0029, 0x1120), (0x0029, 0x1220)]:
      # Access the SERIES Shadow Data - Siemens proprietary tag
      #
      # Need to keep this tag order, Skyra DICOM data had both 1120
      # and 1220 tags and 1220 did not seem to contain CSA data in it.
      try:
        ptag_ser = parse_csa_header(get_tag(ds, de.tag))
        prot_ser = None
        # "MrProtocol" (VA25) and "MrPhoenixProtocol" (VB13) are special
        # elements that contain protocol parameters; very Siemens specific.
        if ptag_ser.get("MrProtocol", ""):
          prot_ser = parse_protocol(ptag_ser["MrProtocol"])
          ptag_ser["MrProtocol"] = None
        if ptag_ser.get("MrPhoenixProtocol", ""):
          prot_ser = parse_protocol(ptag_ser["MrPhoenixProtocol"])
          ptag_ser["MrPhoenixProtocol"] = None
        # Protocol Info
        value = { "protocol": prot_ser, "shadow": ptag_ser }
      except ValueError:
        pass
    elif not key in dont:
      ##print(key+":"+str(de.value))
      if de.VR == "SQ":
        value = []
        for si in de.value:
          value.append(dicom_elements(si))
      else:
        if isinstance(de.value,pydicom.multival.MultiValue):
          value = []
          for v in de.value:
            value.append(v)
        elif isinstance(de.value,pydicom.valuerep.PersonName):
          value = str(de.value)
        else:
          value = conv_val(de.value)
    else:
      ##print(key, de.tag, de.value)
      pass
    if value is not None and value != [] and value != {}:
      if key in contents:
        if key in duplicate_keys:
          found = False
          for v in contents[key]:
            if v == value:
              found = True
              break
          if not found:
            contents[key].append(value)
        else:
          if contents[key] != value:
            contents[key] = [contents[key], value]
            duplicate_keys.append(key)
      else:
        contents[key] = value
  return contents

def read_dicom(filename):
  # Read dicom file and also parse Siemens headers
  dicom = pydicom.read_file(filename)
  info = dicom_elements(dicom)
  info["filename"] = filename
  return dicom, info

def get_tag(dataset, tag, default=None):
  # Get tag from dataset, with default
  if tag not in dataset:
    return default
  if dataset[tag].value == '':
    return default
  return dataset[tag].value

def parse_protocol(protocol_data):
  # Parse Siemens protocol header
  protocol_data = bytes(protocol_data, encoding="utf-8")
  start = protocol_data.find(b"### ASCCONV BEGIN")
  end = protocol_data.find(b"### ASCCONV END ###")

  start += len(b"### ASCCONV BEGIN ###")
  protocol_data = protocol_data[start:end]

  lines = protocol_data.split(b'\n')
  lines = lines[1:]

  f = lambda pair: (pair[0].strip().decode("utf-8"), pair[1].strip(b'"\t ').decode("utf-8"))
  lines = [f(line.split(b'=')) for line in lines if line]

  protocol = {}
  for k,v in lines:
    fields = k.split('.')
    protocol = add_to_dict(fields,v,protocol)

  return protocol

def add_to_dict(fs,v,d):
  # Extract and add entry from siemens protocol header to dict
  if fs == []:
    return v
  if fs[0] == "__attribute__":
    return add_to_dict(fs[1:],v,d)
  match = re.fullmatch(r'(.*)\[([0-9]+)\]', fs[0])
  if match:
    if not match.group(1) in d:
      d[match.group(1)] = {} # Not a list, as indices unpredictable
    d[match.group(1)][match.group(2)] = conv_val(v)
  else:
    if len(fs) > 1:
      if fs[0] in d:
        d[fs[0]] = { **d[fs[0]], **add_to_dict(fs[1:],v,d[fs[0]]) }
      else:
        d[fs[0]] = add_to_dict(fs[1:],v,{})
    else:
      d[fs[0]] = conv_val(v)
  return d

def conv_val(v):
  # Helper to convert value
  try:
    num = int(v)
    return num
  except ValueError:
    try:
      num = float(v)
      return num
    except ValueError:
      return v
  except TypeError:
    return v

def parse_csa_header(tag, little_endian = True):
  # Parse Siemens CSA header info
  #
  # Parser based on Grassroots DICOM code - GDCM:
  #  https://sourceforge.net/projects/gdcm/
  # CSAHeader::LoadFromDataElement() inside gdcmCSAHeader.cxx
  #
  # A Siemens CSA header is a mix of binary glop, ASCII, binary masquerading
  # as ASCII, and noise masquerading as signal. It's undocumented, so there's
  # no specification to which to refer.
  # - The data in the tag is a list of elements, each of which contains
  #   zero or more subelements. The subelements can't be further divided
  #   and are either empty or contain a string.
  # - Everything begins on four byte boundaries.
  # - This code will break on big endian data. I don't know if this data
  #   can be big endian, and if that's possible I don't know what flag to
  #   read to indicate that. However, it's easy to pass an endianness flag
  #   to _get_chunks() should the need to parse big endian data arise.
  # - Delimiters are thrown in here and there; they are 0x4d = 77 which is
  #   ASCII 'M' and 0xcd = 205 which has no ASCII representation.
  # - Strings in the data are C-style NULL terminated.

  DELIMITERS = ("M", "\xcd", 0x4d, 0xcd)
  elements = { }

  current = 0

  # The data starts with "SV10" followed by 0x04, 0x03, 0x02, 0x01.
  # It's meaningless to me, so after reading it, I discard it.
  size, chunks = get_chunks(tag, current, "4s4s")
  current += size

  # Get the number of elements in the outer list
  size, chunks = get_chunks(tag, current, "L")
  current += size
  element_count = chunks[0]

  # Eat a delimiter (should be 0x77)
  size, chunks = get_chunks(tag, current, "4s")
  current += size

  for i in range(element_count):
    # Each element looks like this:
    # - (64 bytes) Element name, e.g. ImagedNucleus, NumberOfFrames,
    #   VariableFlipAngleFlag, MrProtocol, etc. Only the data up to the
    #   first 0x00 is important. The rest is helpfully populated with
    #   noise that has enough pattern to make it look like something
    #   other than the garbage that it is.
    # - (4 bytes) VM
    # - (4 bytes) VR
    # - (4 bytes) syngo_dt
    # - (4 bytes) # of subelements in this element (often zero)
    # - (4 bytes) a delimiter (0x4d or 0xcd)
    size, chunks = get_chunks(tag, current, "64s" + "4s" + "4s" + "4s" + "L" + "4s")
    current += size

    name, vm, vr, syngo_dt, subelement_count, delimiter = chunks

    # The subelements hold zero or more strings. Those strings are stored
    # temporarily in the values list.
    values = [ ]

    for j in range(subelement_count):
      # Each subelement looks like this:
      # - (4 x 4 = 16 bytes) Call these four bytes A, B, C and D. For
      #   some strange reason, C is always a delimiter, while A, B and
      #   D are always equal to one another. They represent the length
      #   of the associated data string.
      # - (n bytes) String data, the length of which is defined by
      #   A (and A == B == D).
      # - (m bytes) Padding if length is not an even multiple of four.
      size, chunks = get_chunks(tag, current, "4L")
      current += size

      length = chunks[0]

      # get a chunk-o-stuff, length indicated by code above.
      # Note that length can be 0.
      size, chunks = get_chunks(tag, current, "%ds" % length)
      current += size
      if chunks[0]:
        values.append(chunks[0])

      # If we're not at a 4 byte boundary, move.
      # Clever modulus code below swiped from GDCM
      current += (4 - (length % 4)) % 4

    # The value becomes a single string item (possibly "") or a list of strings
    if len(values) == 0:
      values = ""
    if len(values) == 1:
      values = values[0]

    if (isinstance(values,bytes)):
      values = values.decode('utf-8')
    elif (isinstance(values,list)):
      if isinstance(values[0], bytes):
        values = [v.decode('utf-8') for v in values]

    if values != None and values != "":
      elements[name.decode('utf-8')] = values

  return elements

def scrub(item):
  # Cleanup bytes from CSA header
  if isinstance(item, bytes):
    # Ensure null-termination:
    i = item.find(b'\x00')
    if i != -1:
      item = item[:i]
    return item.strip()
  else:
    return item

def get_chunks(tag, index, format, little_endian=True):
  # Get chunk from CSA header
  format = ('<' if little_endian else '>') + format
  size = struct.calcsize(format)
  if index+size < len(tag):
    chunks = [scrub(item) for item in struct.unpack(format, tag[index:index + size])]
  else:
    raise ValueError("broken chunk")
  return (size, chunks)

if __name__ == '__main__':
  main()
