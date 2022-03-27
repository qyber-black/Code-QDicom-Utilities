#!/usr/bin/env python3
#
# create_dataset.py - QDicom Utilities
# Create dataset for deep learning from dicoms. This is not the most efficient
# code, but works for the purpose and only needs to be called for each dataset
# once. It's also messy across the different datasets, but still serves as a
# record of how we processed them.
#
# SPDX-FileCopyrightText: Copyright (C) 2021-2022 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Usage example:
#
# ./create_dataset.py -s select.json -d scans -o dataset -v
#
# See --help for full arguments
#
# Selection json file format:
# {
#   "dataset": "DATASET",
#   "width": SCALE_WIDTH,
#   "height": SCALE_HEIGHT,
#
#
#   "slices": [
#     ["PATIENT", "SESSION", "SCAN", "SLICE"],
#     ...
#   ],
#   "density": INT,
#   "register": BOOL,
#   "protocols": [ "PROTOCOL1", "PROTOCOL2", ... ],
#   "compute": [ "COMPUTE1", "COMPUTE2", ... ],
#   "tags": [ "TAG1", "TAG2", ... ]
#
#
#   "ref_protocol": "OUR_NAME",
#   "protocol_map": [
#     ["PROSTATEX_NAME1", "OUR_NAME1"],
#     ...
#   ],
#   "mask_size": MASK_SIZE,
#   "skip_patients": [
#     [PROSTATEX-PATIENT1, COMMENT1],
#     ...
#   ]
# }
#
# dataset: reference for dataset sourcer (swansea-pca, prostatex supported;
#          most info for prostatex comes from their csv files)
#
# Dataset generated for swansea-pca according to:
#
# slices: array of selected 2D slices for data set; each slice is specified by
#         patient-id, session-id, protocol, scan, slice
#         (search in dicom data repository path; script finds the group;
#         paitent-id assumed to be unique)
#
# * width,height: resolution
#                 Rescales the data to this size from original slice size.
#                 Should match resolution of original as far as possible (obvouisly
#                 that is not doable if protocols have different resolutions).
# * density: sampling densities in samples per mm per axis, used to resample stacks, if needed; default 25.
#            We resample slices to the reference slice via the patient coordinate space (and registration
#            if requested). Sampling is done via intersecting the reference slice voxel mapped onto the
#            stack and sampling the overlap per voxel in the reference slice. The voxel value is the
#            weighted average of the reference slice voxel values using the sampled overlap percentage.
#            Samples are taken using a sobol sequence for even coverage of sapce. This is slow (but
#            nearest neighbour or some interpolation scheme may have more accuracy issues). We may change
#            this to something better eventually. For now it works for what we need it for.
# * regsiter: true/false - register stack to reference stack (determined by slices specified);
#             this needs more work.
# * protocols: list of protocols to select
#              Data acquisition protocols: t1-tra, t1-sag,
#                                          t2-cor, t2-sag, t2-tra
#                                          dwi
#              Assumes there is only one corresponding slice in the session
# * compute: computed outputs:
#              adc      - linear adc match with residual
#              adc_q    - quadratic adc match with kurtosis and residual
#              dwi_c-X  - high B-field computed dwi for B=X for linear adc
#              dwi_cq-X - high B-field computed dwi for B=X for quadratic adc
# * tags: produces masks for specified tags; can specify extended tag with ":" such
#         as "suspicious:ext" for exact match or just a tag with a ":" extension,
#         which matches all tags, independent of the extension.
#
# Dataset generated for prostatex according to:
#
# * width,height: resolution
#                 Rescales the data to this size from original slice size.
#                 Should match resolution of original as far as possible (obvouisly
#                 that is not doable if protocols have different resolutions).
# * skip_patient: list of lists with two entires [PATIENT_ID, COMMENT] of patients to ignore
# * ref_protocol: name of the protocol used as reference slice for finding
# * mask_size:    size of square mask for label (in reference slice voxels)
# * protocol_map: list of lists with two entries [ORIG_PROTOCOL, NAME] to map ProstateX
#                 protocols onto ours and select which ones to use

import os
import argparse
import glob
import json
import math
import numpy as np
import sobol_seq
from PIL import Image, ImageDraw
from skimage.draw import polygon, line
import scipy.ndimage as ndi

from read_dicom_siemens import read_dicom

def main():
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', type=str, help='Folder with the dicom files (structure: GROUP/PATIENT/SESSION/SCAN/PATIENT-SESSION-SCAN-SLICE.IMA')
  parser.add_argument('-o', '--out', type=str, help='Folder to store dataset (stores as REF_PATIENT-REF_SESSION/REF_PATIENT-REF_SESSION-REF_SCAN-REF_SLICE-PROTOCOL in npy; png only for display; REF_* variables are taken from the json selection file, while protocol is taken from the actual stack it has been extracted from via the reference slice)')
  parser.add_argument('-s', '--select', type=str, help='Reference slice selection json file (see comments in file for format)')
  parser.add_argument('-p', '--disable-parallel', action='store_true', help='Do not execute in parallel (disable for testing, etc)')
  parser.add_argument('-c', '--check', action='store_true', help='Check if data is different in existing files and report; if force is set, will only overwrite if files are different')
  parser.add_argument('-f', '--force', action='store_true', help='Force processing files, even if target already exists (only overwrites; does not delete); if check is set, will only overwrite if files are different')
  parser.add_argument('-v', '--verbose', action='count', help='Increase output verbosity', default=0)
  args = parser.parse_args()

  # Check arguments
  if args.data == None or not os.path.isdir(args.data):
    print('Dicom folder not specified or is not a directory: %s' % args.data)
    exit(1)
  if args.select == None or not os.path.isfile(args.select):
    print('Selection file not specified or does not exist: %s' % args.select)
    exit(1)
  if args.out == None:
    print('Dataset folder not specified')
    exit(1)

  # Load selection and input/output specification
  f = open(args.select)
  sel_js = json.loads(f.read())
  if "density" not in sel_js:
    sel_js["density"] = 25 # density default (should be integer)
  f.close()

  if sel_js["dataset"] == "swansea-pca":
    swansea_pca(args, sel_js)
  elif sel_js["dataset"] == "prostatex":
    prostatex(args, sel_js)
  else:
    raise Exception(f"Unkown daataset {sel_js['dataset']}")

def swansea_pca(args, sel_js):
  # Collect slices for the same patient-session for swansea-pca
  patient = {}
  for slice in sel_js['slices']:
    ps = slice[0] + "-" + slice[1]
    # Find slice group
    group = None
    for g in os.listdir(args.data):
      dir = os.path.join(args.data,g)
      if os.path.isdir(dir) and os.path.isdir(os.path.join(dir,slice[0])) \
         and os.path.isdir(os.path.join(dir,slice[0],slice[1])):
        group = g
        break
    if group is None:
      raise Exception(f"Patient-session {ps} not found")
    # Add to patient-session
    if ps not in patient:
      patient[ps] = []
    patient[ps].append([group,slice[0],slice[1],slice[2],slice[3]])

  # Process all patient-session collections
  density = int(sel_js["density"])
  if args.disable_parallel:
    for ps in patient:
      process_patient(args.data, patient[ps],ps,sel_js,density,args.out,args.force,args.check,args.verbose)
  else:
    import joblib
    d_inp = joblib.Parallel(n_jobs=-1, prefer="processes", verbose=10*args.verbose) \
              (joblib.delayed(process_patient)(args.data,patient[ps],ps,sel_js,
                                               density,args.out,args.force,args.check,-1)
                for ps in patient)

def process_patient(data, patient, ps, sel_js, density, out, force, check, verbose):
  # Process all slices for the patient
  if verbose > 0:
    print(f"# {ps}")
  elif verbose == -1: # parallel
    print(f"Starting {ps}")

  # Load protocol stacks
  protocols = sel_js['protocols']
  if 'compute' in sel_js: # add anything missing for computed outputs
    for c in sel_js['compute']:
      if (c[0:3] == "adc" or c[0:5] == "dwi_c") and "dwi" not in protocols:
        protocols.append("dwi")
  stacks = get_stacks(os.path.join(data, patient[0][0], patient[0][1], patient[0][2]),
                      protocols, patient[0][3], verbose)

  # Register
  if 'register' in sel_js and sel_js['register']:
    # Find protocol stack for registration; taken from reference slices
    # (reference stack should always be loaded above!)
    ref_protocol = None
    for s in stacks:
      if stacks[s]['scan'] == patient[0][3]:
        ref_protocol = s
        break
    if ref_protocol is None:
      # Registration fixed/reference stack missing
      # - should not happen, as we try to ensure to load the reference stack.
      # We assume we use the same reference stack for all slices in the patient;
      # we load it if it is not there, but separate reference stacks make no sense here,
      # in particular if we register the stacks.
      raise Exception("No protocol for registration found")
    # Register stacks to reference  stack
    register_stacks(stacks, ref_protocol, verbose)

  # For each selected reference slice:
  for ref_slice in patient:
    out_dir = os.path.join(out, ps)
    if not os.path.isdir(out_dir):
      os.makedirs(out_dir)
    # Load reference slice
    _, _, ref_transf_slice2patient, ref_size = get_slice(os.path.join(data, *ref_slice[0:4]),
                                                         ref_slice[4], verbose)
    if ref_transf_slice2patient is None:
      raise Exception("Reference slice {ref_slice} not found")
    # Create dataset files for slice
    for p in sel_js['protocols']:
      # Each channel determined as full 2D area from protocol
      if verbose > 0:
        print(f"  {p}")
      create_slice(out_dir, stacks[p]["data"], stacks[p]["first_slice"], stacks[p]['scan'],
                   stacks[p]["info"], stacks[p]["transf"], p, "",
                   ref_slice[1], ref_slice[2], ref_slice[3], sel_js['width'], sel_js['height'],
                   int(ref_slice[3]), int(ref_slice[4]), ref_transf_slice2patient, ref_size,
                   density, force, check, verbose)
    if 'compute' in sel_js:
      compute_slice(out_dir, sel_js['compute'],
                    ref_slice[1], ref_slice[2], ref_slice[3], sel_js['width'], sel_js['height'], int(ref_slice[4]),
                    ref_transf_slice2patient, ref_size, density,
                    stacks[p]['info']['BitsStored'],
                    force, check, verbose)
    if 'tags' in sel_js:
      generate_masks(out_dir, stacks,
                     sel_js['tags'], os.path.join(data, patient[0][0], patient[0][1], patient[0][2]),
                     ref_slice[1], ref_slice[2], ref_slice[3], sel_js['width'], sel_js['height'], int(ref_slice[4]),
                     ref_transf_slice2patient, ref_size, density,
                     force, check, verbose)
    # Create slice matrix
    if verbose > 0:
      print("  Slice matrix")
    sns = sorted(glob.glob(os.path.join(out_dir,f"{ps}-{ref_slice[3]}-{ref_slice[4]}-*.png")))
    img = {}
    for l  in range(0,len(sns)):
      proto = sns[l].split("-")[3:]
      proto[-1] = proto[-1].split(".")[0]
      k = "-".join(proto)
      if proto[-1] != "all":
        img[k] = Image.open(sns[l]).point(lambda p: p*0.00390630960555428397, mode='RGB').convert('RGB')
    keys = sorted(img.keys())
    l = len(keys)
    img_w = sel_js['width']+4
    img_h = sel_js['height']+4
    img_matrix = Image.new('RGB', (l*img_w,l*img_h))
    sz = 40
    for r in range(0,l):
      for c in range(0,l):
        pos = (c*img_w+2,r*img_h+2)
        if r == c:
          img_matrix.paste(img[keys[r]],pos)
        elif r > c:
          h = Image.new("RGB", img[keys[r]].size, "black")
          rr,_,bb = h.split()
          _,gg,_ = img[keys[r]].split()
          h = Image.merge('RGB',(rr,gg,bb))
          img_matrix.paste(Image.blend(h,img[keys[c]],alpha=0.5), pos)
        else:
          mask = Image.new("L", img[keys[r]].size, "black")
          dr = ImageDraw.Draw(mask)
          for x in range(0,mask.size[0]//sz+1):
            for y in range(0,mask.size[1]//sz+1):
              if (x+y) % 2 == 0:
                dr.rectangle([(x*sz,y*sz),(sz+x*sz,sz+y*sz)], fill="white", outline=128)
          img_matrix.paste(Image.composite(img[keys[r]],img[keys[c]],mask), pos)
    dr = ImageDraw.Draw(img_matrix)
    for x in range(0,img_matrix.size[0]//img_w+1):
      dr.line([(x*img_w-1,0),(x*img_w-1,img_matrix.size[1])],width=2,fill=(96,96,128))
    for y in range(0,img_matrix.size[1]//img_h+1):
      dr.line([(0,y*img_h-1),(img_matrix.size[0],y*img_h-1)],width=2,fill=(96,96,128))
    img_matrix.save(os.path.join(out_dir,f"{ps}-{ref_slice[3]}-{ref_slice[4]}-all.png"), optimize=True)
  if verbose == -1: # parallel
    print(f"Stopping {ps}")

def get_slice(dir,slice,verbose):
  path, scan = os.path.split(dir)
  path, session = os.path.split(path)
  path, patient = os.path.split(path)
  path, group = os.path.split(path)
  if verbose > 0:
    print(f"  {group}-{patient}-{session}-{scan}: {slice}")
  # Load reference slice
  file0 = os.path.join(dir,f"{patient}-{session}-{scan}-{slice}.IMA")
  if not os.path.isfile(file0):
    return None, None, None, None
  # 2nd slice for 3D reference
  file1 = os.path.join(dir,f"{patient}-{session}-{scan}-{int(slice)+1:04d}.IMA")
  if os.path.isfile(file1):
    # Found 2nd reference
    zdir = 1.0
  else:
    file1 = os.path.join(dir,f"{patient}-{session}-{scan}-{int(slice)-1:04d}.IMA")
    if os.path.isfile(file1):
      # Found 2nd reference, but before ref. slice
      zdir = -1.0
    else:
      # No 3D reference
      zdir = 0.0

  # Get reference frame in patient space
  dicom0, info0 = read_dicom(file0)
  # Defaults:
  S = np.float64([0,0,0]) # Image position (top-left-center voxel pos.)
  X = np.float64([1,0,0]) # X direction
  Y = np.float64([0,1,0]) # Y direction
  Z = np.float64([0,0,1]) # Z direction
  D = np.float64([1,1,1]) # X, Y, Z spacing
  if "ImagePositionPatient" in info0:
    S = np.float64([info0["ImagePositionPatient"][0],
                    info0["ImagePositionPatient"][1],
                    info0["ImagePositionPatient"][2]])
  if "ImageOrientationPatient" in info0:
    X = np.float64([info0["ImageOrientationPatient"][0],
                    info0["ImageOrientationPatient"][1],
                    info0["ImageOrientationPatient"][2]])
    Y = np.float64([info0["ImageOrientationPatient"][3],
                    info0["ImageOrientationPatient"][4],
                    info0["ImageOrientationPatient"][5]])
  if "PixelSpacing" in info0:
    D[0:2] = np.float64([info0["PixelSpacing"][0],
                         info0["PixelSpacing"][1]])
  if "SliceThickness" in info0:
    D[2] = np.float64(info0["SliceThickness"])
  Z = np.cross(X,Y)
  Z = Z / np.linalg.norm(Z) * D[2]
  if zdir != 0.0:
    _, info1 = read_dicom(file1)
    if "ImagePositionPatient" in info1:
      S1 = np.float64([info1["ImagePositionPatient"][0],
                       info1["ImagePositionPatient"][1],
                       info1["ImagePositionPatient"][2]])
      Z = zdir * (S1 - S)

  # 3D MRI image plane conversion via two slices in the stack; that means we
  # load tghe dicoms a few times, as we do not reuse the stack info, but fast
  # enough here.
  # Matrix to convert from image index (with slice index relative to current slice)
  # to patient coordinates in mm
  transf_slice2patient = np.matrix([[X[0]*D[0], Y[0]*D[1], Z[0], S[0]],
                                    [X[1]*D[0], Y[1]*D[1], Z[1], S[1]],
                                    [X[2]*D[0], Y[2]*D[1], Z[2], S[2]],
                                    [      0.0,       0.0,  0.0, 1.0]],
                                   dtype=np.float64)
  try:
    size = dicom0.pixel_array.shape
  except:
    size = None
  if verbose > 0:
    print("    Transf stack to patient", ('\n' + str(transf_slice2patient)).replace('\n','\n      '))
    print(f"    Size: {size}")
  return dicom0, info0, transf_slice2patient, size

def get_stacks(patient_base, protocols, ref_scan, verbose):
  # Load dicom stacks for protocols
  stacks = {}
  # Parse stacks
  for scan in sorted(os.listdir(patient_base)):
    path = os.path.join(patient_base,scan)
    if os.path.isdir(path):
      # Find protocol
      sns = sorted(glob.glob(os.path.join(path,'*.IMA')))
      if len(sns) > 0:
        slice = os.path.basename(sns[0]).split("-")[3].split(".")[0]
        dicom, info, transf_stack2patient, _ = get_slice(path, slice, 0)
        first_slice = int(slice)
        if '[CSA Series Header Info]' in info and 'protocol' in info['[CSA Series Header Info]'] and \
           'tProtocolName' in info['[CSA Series Header Info]']['protocol']:
          org_protocol = info['[CSA Series Header Info]']['protocol']['tProtocolName']
          protocol = None
          # Check protocol
          if (org_protocol == 'localizer' or # Ignoring localizer
              org_protocol[0:12] == 't1_vibe_tra_' or # Ignore contrast agent uptake
              org_protocol[0:6] == 'csi3d_' or # Ignoring spectroscopy
              org_protocol[0:3] == 'AX '): # Ignoring unknown protocol
            protocol = None
          elif org_protocol[0:8] == 't1_axial' or org_protocol[0:10] == 't1_tse_tra' or org_protocol[0:11] == 't1_fl2d_tra':
            protocol = 't1-tra'
          elif org_protocol[0:10] == 't1_tse_sag':
            protocol = 't1-sag'
          elif org_protocol[0:8] == 't2_axial' or org_protocol[0:6] == 't2_tra' or org_protocol[0:10] == 't2_tse_tra':
            protocol = 't2-tra'
          elif org_protocol[0:6] == 't2_cor' or org_protocol[0:10] == 't2_tse_cor':
            protocol = 't2-cor'
          elif org_protocol[0:6] == 't2_sag' or org_protocol[0:10] == 't2_tse_sag':
            protocol = 't2-sag'
          elif org_protocol[0:3] == 'dwi' or org_protocol[0:4] == 'ep2d':
            protocol = "dwi"
            if 'ImageType' in info and ('ADC' in info['ImageType'] or # Ignoring ADC
                                        'CALC_BVALUE' in info['ImageType']): # Ignoring calculated high b-value
              protocol = None
          else:
            raise Exception(f"Unkown protocol {org_protocol}")
          # Ensure we load reference stack and it has a protocol
          if ref_scan == scan:
            if protocol == None:
              raise Exception("Unknown protocol for refernece slice stack")
          if protocol in stacks:
            # Check if duplicate (we always pick the first, but warn)
            print(f"Warning: duplicate {protocol} in scan {scan} in {patient_base}")
          elif protocol is not None and (protocol in protocols or ref_scan == scan):
            # Found the requested protocol (or its the reference slice stack) - load stack
            if verbose > 0:
              print(f"  Found {protocol} ({scan}/{org_protocol})")
            data = [dicom.pixel_array.astype(np.uint16, order='C', casting='safe', copy=False)]
            X0 = transf_stack2patient[0:3,3]/transf_stack2patient[3,3]
            Z0 = transf_stack2patient[0:3,2]
            max_diff = np.linalg.norm(0.25 * Z0)
            X1 = X0 + Z0
            for k in range(1,len(sns)):
              slice = os.path.basename(sns[k]).split("-")[3].split(".")[0]
              dicom, _, transf, _ = get_slice(path, slice, 0)
              # Check if step size is consistent with slice numbers
              # We assume dwi is OK (as pos reset between different b values makes it harder to check and should not be necessary really; just for safety)
              if protocol != "dwi":
                XX = transf[0:3,3]/transf[3,3]
                diff = np.linalg.norm(XX-X1)
                if diff > max_diff:
                  raise Exception(f"Steps between slice not consistent with step number: {diff} for {slice}")
                X1 += Z0
              data.append(dicom.pixel_array.astype(np.uint16, order='C', casting='safe', copy=False))
            # dicoms sorted by file order / number in swansea-pca set
            stacks[protocol] = {
                'data': np.stack(data),
                'info': info,
                'scan': scan,
                'first_slice': first_slice,
                'transf': transf_stack2patient
              }
  for p in protocols:
    if p not in stacks:
      raise Exception(f"Protocol {p} not found for {patient_base}")
  return stacks

def create_slice(output, stack, first_slice, stack_scan, info, transf_stack2patient, protocol, protocol_ext,
                 patient, session, scan, width, height, ref_slice_scan,
                 ref_slice_num, ref_transf_slice2patient, ref_size,
                 density, force, check, verbose):
  # Create specific channel slice in dataset from reference slice

  # Bits known?
  if not "BitsStored" in info:
    raise Exception("No BitsStored information")
  # Greyscale?
  if info["SamplesPerPixel"] != 1:
    raise Exception("Not single intensity")

  if protocol == "dwi":
    # Separate dwi acquisitions
    if "sDiffusion" in info['[CSA Series Header Info]']['protocol'] and \
       "alBValue" in info['[CSA Series Header Info]']["protocol"]["sDiffusion"]:
      # Extract stacks for each b-value
      b = {"0": "0"} # 0 b-field not in list (but below will overwrite if it is, anyway)
      keys = ["0"]
      for idx in info['[CSA Series Header Info]']['protocol']['sDiffusion']['alBValue']:
        try:
          a = int(idx) # if not a number, ignore
          b[idx] = info['[CSA Series Header Info]']['protocol']['sDiffusion']['alBValue'][idx]
          if idx not in keys:
            keys.append(idx)
        except:
          pass
      n_slices = stack.shape[0]
      if (n_slices // len(keys)) * len(keys) != n_slices:
        raise Exception(f"Number of B-field values mismatch: {n_slices} slices for {len(keys)} keys - {keys}")
      per_stack = n_slices // len(keys)
      for k in range(0,len(keys)):
        # Filename - refers to reference scan, but indicates stack by protocol
        fnb = f"{patient}-{session}-{scan}-{ref_slice_num:04d}-{protocol}{protocol_ext}-{int(b[keys[k]]):04d}"
        # Check if it exists
        if not force and not check and os.path.isfile(os.path.join(output,fnb+".npy")) and \
           os.path.isfile(os.path.join(output,fnb+".png")):
          pass
        else:
          if verbose > 0:
            print(f"    B={b[keys[k]]}")
          # Cutout slice from b-stack dwi data
          slice = extract_slice(stack[k*per_stack:(k+1)*per_stack,:,:],
                                transf_stack2patient, width, height,
                                ref_transf_slice2patient, ref_size,
                                ref_slice_scan, ref_slice_num, stack_scan, first_slice,
                                density, verbose)
          slice /= np.float64(2**info['BitsStored']-1) # Normalise
          # Save result
          save_slice(output, fnb, slice, force, check, verbose)
    else:
      raise Exception("No b-field values for DWI")
  else:
    # Filename - refers to reference scan, but indicates stack by protocol
    fnb = f"{patient}-{session}-{scan}-{ref_slice_num:04d}-{protocol}{protocol_ext}"
    # Check if it exists
    if not force and not check and os.path.isfile(os.path.join(output,fnb+".npy")) and \
       os.path.isfile(os.path.join(output,fnb+".png")):
      return
    # Cutout slice from protocol data
    slice = extract_slice(stack, transf_stack2patient, width, height,
                          ref_transf_slice2patient, ref_size,
                          ref_slice_scan, ref_slice_num, stack_scan, first_slice,
                          density, verbose)
    slice /= np.float64(2**info['BitsStored']-1) # Normalise
    # Save result
    save_slice(output, fnb, slice, force, check, verbose)

def extract_slice(stack, transf_stack2patient, width, height, ref_transf_slice2patient,
                  ref_size, ref_slice_scan, ref_slice_num, scan, first_slice, density, verbose):
  # We are extracting from the reference slice at same resolution, so just copy it out
  if int(ref_slice_scan) == int(scan) and ref_size == (width,height):
    if verbose > 0:
      print("    Using reference slice without resampling")
    return stack[ref_slice_num-first_slice,0:height,0:width].astype(np.float64)
  # Transformation to match slice resolution to desired resolution
  x_scale = np.float64(ref_size[1])/np.float64(width)
  y_scale = np.float64(ref_size[0])/np.float64(height)
  transf_slice = np.matrix([[x_scale,0.0,0.0,0.0],
                            [0.0,y_scale,0.0,0.0],
                            [0.0,0.0,1.0,0.0],
                            [0.0,0.0,0.0,1.0]],dtype=np.float64)
  # (Reference) slice to (data) stack transformations
  transf_ref2stack = np.linalg.inv(transf_stack2patient) @ ref_transf_slice2patient @ transf_slice
  transf_stack2ref = np.linalg.inv(ref_transf_slice2patient @ transf_slice) @ transf_stack2patient
  # Slice
  slice = np.zeros((height,width), dtype=np.float64)

  # Intersections of Voxels is hard, even if doable, even with exact arithmetic:
  #   J.-P. Reveilles. The Geometry of the Intersection of Voxel Spaces.
  #   Electronic Notes in Theoretical Computer Science, 46:285-308, 2001.
  # Instead, we approximate the overlap per voxel sampling the volume using a Sobol sequence

  # Determine number of sample points based on density
  p = transf_ref2stack @ np.matrix([[0.0, 1.0, 1.0, 0.0, 0.0,1.0,1.0,0.0],
                                    [0.0, 0.0, 1.0, 1.0, 0.0,0.0,1.0,1.0],
                                    [-0.5,-0.5,-0.5,-0.5,0.5,0.5,0.5,0.5],
                                    [ 1.0, 1.0, 1.0, 1.0,1.0,1.0,1.0,1.0]],
                                    dtype=np.float64)
  p = np.divide(p[0:3,:],np.vstack((p[3,:],p[3,:],p[3,:])))
  n_samples = int(np.ceil(np.prod(np.amax(p, axis=1) - np.amin(p, axis=1)) * density * density * density))
  if verbose > 0:
    print(f"    Resampling with {n_samples} per resampled/output voxel")

  # Generate samples inside a stack voxel, mapped to the slice ("directions" only, centered at 0)
  skip = math.floor(math.log(n_samples*3,2))
  samples = np.transpose(sobol_seq.i4_sobol_generate(3,n_samples+skip)[skip:,:]) - np.matrix([[0.0],[0.0],[0.5]])
  samples = transf_stack2ref @ np.vstack((samples,np.zeros((1,n_samples)))) # Note, these are directions from the voxe position

  # For each voxel in the slice (could also do for each stack voxel, but likely more unused voxels)
  for x_idx in range(0,width):
    for y_idx in range(0,height):
      # Interesect ref-slice voxels with stack voxels to determine slice voxel value
      # Ref-slice voxel to stack voxel
      p = transf_ref2stack @ np.matrix([[x_idx,x_idx+1,x_idx+1,x_idx,  x_idx,x_idx+1,x_idx+1,x_idx],
                                        [y_idx,y_idx  ,y_idx+1,y_idx+1,y_idx,y_idx,  y_idx+1,y_idx+1],
                                        [-0.5, -0.5,   -0.5,   -0.5,   0.5,  0.5,    0.5,    0.5],
                                        [ 1.0,  1.0,    1.0,    1.0,   1.0,  1.0,    1.0,    1.0]],
                                       dtype=np.float64)
      p = np.divide(p[0:3,:],np.vstack((p[3,:],p[3,:],p[3,:])))
      p0 = np.amin(p, axis=1)
      p1 = np.amax(p, axis=1)
      # Index range of stack voxels to consider
      p_idx = np.int64(np.hstack((np.floor(p0), np.ceil(p1)+1)))
      np.clip(p_idx[0,:], 0, stack.shape[2], p_idx[0,:])
      np.clip(p_idx[1,:], 0, stack.shape[1], p_idx[1,:])
      np.clip(p_idx[2,:], 0, stack.shape[0], p_idx[2,:])
      # Integrate over stack voxels involved and intersesct each with slice voxel by sampling
      v = np.float64(0.0)
      for vx_idx in range(p_idx[0,0],p_idx[0,1]):
        for vy_idx in range(p_idx[1,0],p_idx[1,1]):
          for vz_idx in range(p_idx[2,0],p_idx[2,1]):
            q = (transf_stack2ref @ np.matrix([[vx_idx],[vy_idx],[vz_idx],[1.0]],dtype=np.float64)) + samples
            q = np.divide(q[0:3,:],np.vstack((q[3,:],q[3,:],q[3,:]))) - np.matrix([[x_idx],[y_idx],[-0.5]],dtype=np.float64)
            c = np.sum(np.sum(np.abs(np.floor(q)), axis=0) == 0)
            v += c / np.float64(n_samples) * stack[vz_idx,vy_idx,vx_idx]
      slice[y_idx,x_idx] = v
  return slice

def compute_slice(output, compute, patient, session, scan, width, height, ref_slice_num,
                  ref_transf_slice2patient, ref_size, density, bits, force, check, verbose):
  # Compute slices from those extracted (from dwi data) - ADC and high b-value DWI

  # Max value from number of bits
  max_value = np.float64(2**bits-1)

  # Load DWI data
  dwi_files = sorted(glob.glob(os.path.join(output,f"{patient}-{session}-{scan}-{ref_slice_num:04d}-dwi-*.npy")))
  dwi = [None]*len(dwi_files)
  bfield = [None]*len(dwi_files)
  for k in range(0,len(dwi_files)):
    bfield[k] = int(dwi_files[k].split("-")[-1].split(".")[0])
    dwi[k] = np.load(dwi_files[k])

  # Sort B values
  bfield_idx = np.argsort(bfield)
  bfield_srt = [bfield[b] for b in bfield_idx]

  # Compute ADC and related for slice
  if verbose > 0:
    print(f"  Compute {compute} for {patient}-{session}-{ref_slice_num:04d}")
  rs, cs = dwi[bfield_idx[0]].shape
  # Prepare requested maps - treat NaN values (matching failed) as 0 to avoid processing problems later on
  if "adc" in compute:
    adc = np.zeros((rs,cs))
    adcr = np.zeros((rs,cs))
  if "adc_q" in compute:
    adcq = np.zeros((rs,cs))
    adcqr = np.zeros((rs,cs))
    kurtosis = np.zeros((rs,cs))
  dwic = {}
  dwicq = {}
  for c in compute:
    if c[0:6] == "dwi_c-":
      X = c.split("-")[1]
      dwic[X] = np.zeros((rs,cs))
    if c[0:7] == "dwi_cq-":
      X = c.split("-")[1]
      dwicq[X] = np.zeros((rs,cs))
  for r in range(0,rs):
    for c in range(0,cs):
      # Collect DWI values across B values for slices
      # Scale [0,1] dwi values to 12bit, for consistency.
      with np.errstate(divide='ignore'):
        S = np.log(np.asarray([dwi[b][r,c] for b in bfield_idx], dtype=np.float64) * max_value)
      # Linear ADC fit: ln(S(b)) = ln(S(0)) - b ADC
      fitl, _, frank, _, _ = np.polyfit(bfield_srt,S,1,full=True)
      if frank == 2 and fitl[0] < 0:
        if 'adc' in locals():
          adc[r,c] = -1e6 * fitl[0]
          adcr[r,c] = np.linalg.norm(np.polyval(fitl,bfield_srt)-S)
        if len(dwic) > 0:
          for k in dwic:
            dwic[k][r,c] = np.exp(np.polyval(fitl,int(k)))
      # Kurtosis fit: ln(S(b)) = ln(S(0)) - b ADC + 1/6 b^2 ADC^2 KURTOSIS
      fitq, _, frank, _, _ = np.polyfit(bfield_srt,S,2,full=True)
      if frank == 3 and fitq[1] < 0 and fitq[0] > 0:
        if 'adcq' in locals():
          adcq[r,c] = -1e6 * fitq[1]
          adcqr[r,c] = np.linalg.norm(np.polyval(fitq,bfield_srt)-S)
          q = fitq[1] * fitq[1]
          if q > 1e-15:
            kurtosis[r,c] = 6.0 * fitq[0] / q
        if len(dwicq) > 0:
          for k in dwicq:
            dwicq[k][r,c] = np.exp(np.polyval(fitq,int(k)))
  if 'adc' in locals():
    # Normalise according to dicom range
    adc /= max_value
    adcr /= max_value
    save_slice(output, f"{patient}-{session}-{scan}-{ref_slice_num:04d}-adc", adc, force, check, verbose)
    save_slice(output, f"{patient}-{session}-{scan}-{ref_slice_num:04d}-adc_r", adcr, force, check, verbose)
  for k in dwic:
    # Normalise according to dicom range
    dwic[k] /= max_value
    save_slice(output, f"{patient}-{session}-{scan}-{ref_slice_num:04d}-dwi_c-{int(k):04d}", dwic[k], force, check, verbose)
  if 'adcq' in locals():
    # Normalise according to dicom range
    adcq /= max_value
    kurtosis /= max_value
    adcqr /= max_value
    save_slice(output, f"{patient}-{session}-{scan}-{ref_slice_num:04d}-adc_q", adcq, force, check, verbose)
    save_slice(output, f"{patient}-{session}-{scan}-{ref_slice_num:04d}-adc_qk", kurtosis, force, check, verbose)
    save_slice(output, f"{patient}-{session}-{scan}-{ref_slice_num:04d}-adc_qr", adcqr, force, check, verbose)
  for k in dwicq:
    # Normalise according to dicom range
    dwicq[k] /= max_value
    save_slice(output, f"{patient}-{session}-{scan}-{ref_slice_num:04d}-dwi_qc-{int(k):04d}", dwicq[k], force, check, verbose)

def generate_masks(output, stacks, tags, patient_base,
                   patient, session, scan, width, height, ref_slice_num,
                   ref_transf_slice2patient, ref_size, density, force, check, verbose):
  # Create masks for specified tags
  tags = [t.lower() for t in tags]

  # Load annoations
  if verbose > 0:
    print("  Loading annotations")
  a_files = sorted(glob.glob(os.path.join(patient_base,'*',f"{patient}-{session}-*_annotations.json")))
  rois = {}
  for k in range(0,len(a_files)):
    f = open(a_files[k])
    anno = json.loads(f.read())
    f.close()
    if 'polygons' in anno:
      for p in anno['polygons']:
        for t in tags:
          if ":" in t:
            # If : is in tag, we want extended tag (id of annotator, etc)
            # so an exact match.
            ptag = p['tag'].lower()
          else:
            # If : is not in tag, we accept any tag matching part before ":" (if there is)
            ptag = p['tag'].lower().split(":")[0]
          if ptag == t:
            if not ptag in rois:
              rois[ptag] = []
            rois[ptag].append({"x":p['x'], "y":p['y'], "slice":p['slice']})
            if verbose > 0:
              print(f"    {ptag} - {p['slice']}")

  x_scale = np.float64(width)/np.float64(ref_size[1])
  y_scale = np.float64(height)/np.float64(ref_size[0])
  transf_slice = np.matrix([[x_scale,0.0,0.0,0.0],
                            [0.0,y_scale,0.0,0.0],
                            [0.0,0.0,1.0,0.0],
                            [0.0,0.0,0.0,1.0]],dtype=np.float64)
  ref_inv = transf_slice * np.linalg.inv(ref_transf_slice2patient)

  for t in rois:
    if verbose > 0:
      print(f"  {t} mask")
    mask_slice = np.zeros((height,width))
    for p in rois[t]:
      fns = p['slice'].split('-')[2:4]
      _, _, tag_transf_slice2patient, tag_size = get_slice(os.path.join(patient_base, fns[0]),
                                                           fns[1], 0)
      # Apply registration, if we have it (otherwise we ignore, even if reg. requested!)
      for s in stacks:
        if stacks[s]['scan'] == fns[0]:
          if 'transf_reg' in stacks[s]:
            tag_transf_slice2patient *= stacks[p]["transf_reg"]
      # Convert polygon coordinates
      draw_roi(p['x'],p['y'],mask_slice,ref_inv * tag_transf_slice2patient)
    if np.sum(mask_slice) > 0: # Only store if we have any region at all
      save_slice(output, f"{patient}-{session}-{scan}-{ref_slice_num:04d}-{t}", mask_slice, force, check, verbose)

def draw_roi(rx,ry,mask,transf):
  # Draw region of interest in mask; intersect with slice at z=0
  rx.append(rx[0])
  ry.append(ry[0])
  xs = []
  ys = []
  a = None
  # Find sequence of x,y coordinates to define the polygon in the slice
  for x,y in zip(rx,ry):
    pos = transf * np.matrix([[np.float64(x)],[np.float64(y)],[0],[1]])
    pos = np.matrix([[pos[0,0]],[pos[1,0]],[pos[2,0]]])/pos[3,0]
    if not isinstance(a,np.matrix): # First point
      a = pos
    else: # Next point
      # Intersect with slice at z=0
      # t1/2 = ([0,0,+/-0.5]-a).[0;0;1] / ((b-a).[0;0;1])
      div = pos[2,0] - a[2,0]
      if np.abs(div) >= 1e-6: # Intersection
        t1 = (-0.5-a[2,0]) / div
        t2 = (0.5-a[2,0]) / div
        if t2 < t1: # Order intersection s.t. t1, then t2, from a
          h = t1
          t1 = t2
          t2 = h
        p1 = a + t1 * (pos-a)
        p2 = a + t2 * (pos-a)
        if t1 >= 0: # a ...
          if t1 <= 1: # a p1 ...
            xs.append(np.round(p1[0,0]))
            ys.append(np.round(p1[1,0]))
            if t2 <= 1: # a p1 p2 b
              xs.append(np.round(p2[0,0]))
              ys.append(np.round(p2[1,0]))
            # else a p1 b p2
          # else a b p1 p2
        else: # t1 < 0 - p1 ...
          if t2 >= 0: # p1 a ...
            xs.append(np.round(a[0,0]))
            ys.append(np.round(a[1,0]))
            if t2 <= 1: # p1 a p2 b
              xs.append(np.round(p2[0,0]))
              ys.append(np.round(p2[1,0]))
            # else p1 a b p2
          # else p1 p2 a b
      else: # Parallel
        if np.abs(a[2,0]) <= 0.5: # inside
          xs.append(np.round(a[0,0]))
          ys.append(np.round(a[1,0]))
      a = pos
  # Draw the polygon
  s = mask.shape
  if len(xs) > 0:
    rs,cs = polygon(ys,xs,s)
    if len(rs) == 0: # Flat - try to indicate location anyway
      xmax = -1
      xmin = s[1]
      ymax = -1
      ymin = s[0]
      for l in range(0,len(xs)):
        xi = int(xs[l])
        yi = int(ys[l])
        if xi < s[1] and yi < s[0] and xi >= 0 and yi >= 0:
          mask[yi,xi] = 1.0
          if yi < ymin:
            ymin = yi
          if yi > ymax:
            ymax = yi
          if xi < xmin:
            xmin = xi
          if xi > xmax:
            xmax = xi
      if xmin < s[1] and xmax > -1 and ymin < s[0] and ymax > -1:
        rs,cs = line(ymin,xmin,ymax,xmax)
        mask[rs,cs] = 1.0
    else:
      mask[rs,cs] = 1.0

def register_stacks(stacks, ref, verbose):
  import SimpleITK as sitk

  def get_transf(stack):
    # Get transformation from stack to patient space (used as virtual image space for registartion)
    # We need the inverse as the mapping is from the virtual domain to the image domain
    transf = stack["transf"]
    t = sitk.AffineTransform(3)
    t.SetMatrix([np.float64(transf[r,c]) for r in range(0,3) for c in range(0,3)])
    t.SetTranslation([np.float64(transf[r,3]) for r in range(0,3)])
    return t.GetInverse()

  # Reference stack for registration
  if verbose > 0:
    print(f"  Register to {ref}")
  fixed = sitk.GetImageFromArray(np.array(stacks[ref]["data"],dtype=np.float64))
  # Transformation from patient/vritual image space to stack
  transf_f = get_transf(stacks[ref])

  for p in stacks:
    if p != ref:
      if verbose > 0:
        print(f"    Registering {p}")
      if p == "dwi":
        # For dwi, only get first stack from full stack
        # We do not align the different b-field stacks separately
        if "sDiffusion" in stacks[p]['info']['[CSA Series Header Info]']['protocol'] and \
           "alBValue" in stacks[p]['info']['[CSA Series Header Info]']["protocol"]["sDiffusion"]:
          # Extra size field makes this correct
          b_fields = len(stacks[p]['info']['[CSA Series Header Info]']['protocol']['sDiffusion']['alBValue'])
          if "0" in stacks[p]['info']['[CSA Series Header Info]']['protocol']['sDiffusion']['alBValue']:
            b_fields -= 1 # In case 0 b-field is actually in list
          per_stack = stacks[p]['data'].shape[0] // b_fields
          # Get first stack
          moving = sitk.GetImageFromArray(np.array(stacks[p]["data"][0:per_stack,:,:],dtype=np.float64))
        else:
          raise Exception("DWI stack without b-field information")
      else:
        # Get full stack for any other protocol
        moving = sitk.GetImageFromArray(np.array(stacks[p]["data"],dtype=np.float64))
      # Transformation from patient/vritual image space to stack
      transf_m = get_transf(stacks[p])

      # Registration: needs more work - quite poor convergence and neither repeatable nor reliable
      registration = sitk.ImageRegistrationMethod()
      # Similarity metric settings.
      #registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=128)
      registration.SetMetricAsMeanSquares() # Seems most consistent
      registration.SetMetricSamplingStrategy(registration.REGULAR) # To make it more deterministic, instead of RANDOM
      registration.SetMetricSamplingPercentage(0.25)
      registration.SetInterpolator(sitk.sitkLinear)
      # Optimizer settings.
      registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000,
                                                 convergenceMinimumValue=1e-8,
                                                 convergenceWindowSize=20)
      registration.SetOptimizerScalesFromPhysicalShift()
      # Setup for the multi-resolution framework.
      registration.SetShrinkFactorsPerLevel(shrinkFactors=[2,1])
      registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[1,0])
      registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
      # Set the initial moving and optimized transforms.
      registration.SetFixedInitialTransform(transf_f)
      registration.SetMovingInitialTransform(transf_m)
      registration.SetInitialTransform(sitk.Euler3DTransform(), inPlace=False)
      # Need to compose the transformations after registration.
      done=False
      cnt=1
      while not done:
        try:
          cnt += 1
          transf_final = registration.Execute(fixed, moving)
          done = True
        except Exception as e:
          if cnt > 100:
            raise e
          print(f"      retry {cnt}")
          pass

      # Assuming we have a single Euler3DTransform, get it:
      transf_reg = sitk.CompositeTransform(transf_final)
      if transf_reg.GetNumberOfTransforms() > 1:
        raise Exception("More than one transform in registartion transformation")
      transf_reg = sitk.Euler3DTransform(transf_reg.GetNthTransform(0))
      # Get transformation matrix
      A = np.array(transf_reg.GetMatrix()).reshape(3,3)
      c = np.array(transf_reg.GetCenter())
      t = np.array(transf_reg.GetTranslation())
      transf_r = np.eye(4)
      transf_r[0:3,0:3] = A
      transf_r[0:3,3] = -np.dot(A,c)+t+c
      # Augment stack transform
      stacks[p]["transf_reg"] = np.linalg.inv(transf_r) # Used for tag masks
      stacks[p]["transf"] = stacks[p]["transf"] * stacks[p]["transf_reg"]

      # Report
      if verbose > 0:
        norm_tx = np.linalg.norm(transf_r[0:3,0:3])
        norm_tr = np.linalg.norm(transf_r[0:3,3])
        print("      Registration transf", ('\n' + str(transf_r)).replace('\n','\n        '))
        print(f"      Norm - transform: {norm_tx}; translation: {norm_tr}")
        print(f"      Final metric: {registration.GetMetricValue()}")
        print(f"      Stop: {registration.GetOptimizerStopConditionDescription()}")

def prostatex(args, sel_js):
  # Collect slices for the same patient-session for prostatex
  #
  # For now we use the computed slices (ADC, high b-field) only as given in the
  # data and do not compute them (even if DWI is available); some patients in the
  # dataset do not have these available (is reported as missing, if verbose).
  #
  # If protocol types are duplicated, we usually use the first scan (if we know about
  # it for the protocol; see "if more folders, use first match" below).
  import csv
  for group in ["Train", "Test"]:
    # Get Lesion info
    if args.verbose > 0:
      print(f"Loading {group} findings")
    patient = {} # dict mapping patients to list of fiding ids
    label = {} # label[PATIENT][FID] = label of FID
    with open(os.path.join(args.data, 'Lesions', f'ProstateX-Findings-{group}.csv')) as csvfile:
      findings = csv.reader(csvfile, delimiter=',')
      # Process all findings
      for row in findings:
        if row[0] != 'ProxID': # Skip header
          if row[0] not in patient:
            patient[row[0]] = [int(row[1])]
          else:
            patient[row[0]].append(int(row[1]))
          if row[0] not in label:
            label[row[0]] = {}
          if len(row) > 4:
            label[row[0]][row[1]] = "suspicious_sq" if row[4] == "TRUE" else "normal_sq"
          else:
            label[row[0]][row[1]] = "unknown_sq"
    # Remove patients with potential issues
    for p in sel_js["skip_patients"]:
      if p[0] in patient:
        if args.verbose > 2:
          print(f"Skipping patient {p[0]}: {p[1]}")
        del patient[p[0]]

    # Get images info
    slices = {} # slices[PATIENT][FID][NAME] = [i,j,k]
    world_matrix = {} # world_matrix[PATIENT][PROTOCOL][FID] = Matrix
    ser_num = {} # world_matrix[PATIENT][PROTOCOL][FID] = ser_num
    pos = {} # pos[PATIENT][PROTOCOL][FID] = pos of FID
    with open(os.path.join(args.data, 'Lesions', f'ProstateX-Images-{group}.csv')) as csvfile:
      if args.verbose > 0:
        print(f"Loading {group} images")
      images = csv.reader(csvfile, delimiter=',')
      # Process all images
      for row in images:
        if row[0] != 'ProxID': # Sklip header
          use = False
          row[1] = row[1].replace("_","").rstrip("0123456789")
          for pm in sel_js["protocol_map"]:
            if row[1] == pm[0]:
              protocol_name = pm[1]
              use = True
              break
          if use:
            if row[0] not in slices:
              slices[row[0]] = { }
            if row[2] not in slices[row[0]]:
              slices[row[0]][row[2]] = {}
            if row[0] not in world_matrix:
              world_matrix[row[0]] = {}
              pos[row[0]] = {}
              ser_num[row[0]] = {}
            if protocol_name not in world_matrix[row[0]]:
              world_matrix[row[0]][protocol_name] = {}
              pos[row[0]][protocol_name] = {}
              ser_num[row[0]][protocol_name] = {}
            row[2] = int(row[2])
            row[11] = int(row[11])
            if row[2] in world_matrix[row[0]][protocol_name] and \
               ser_num[row[0]][protocol_name][row[2]] < row[11]:
              # We use the first matching protocol, so anything later is ignored
              pass
            else:
              slices[row[0]][str(row[2])][row[1]] = [int(l) for l in row[5].split(' ')]
              world_matrix[row[0]][protocol_name][row[2]] = np.array(row[4].split(",")).astype(np.float64).reshape(4,4)
              ser_num[row[0]][protocol_name][row[2]] = row[11]
              pos[row[0]][protocol_name][row[2]] = np.array((row[3].strip()+" 1").split(" ")).astype(np.float64)
              # Testing matrix/pos/ijk relation
              ijk = np.array((row[5]+" 1").split(" ")).astype(np.float64)
              pos_ijk = np.floor(np.linalg.inv(world_matrix[row[0]][protocol_name][row[2]]) @ \
                                 pos[row[0]][protocol_name][row[2]] + 0.5)
              if np.linalg.norm(ijk-pos_ijk) > 0.0:
                raise Exception(f"{row[0]},{protocol_name},{row[2]} - pos to ijk does not match ijk")
          else:
            if args.verbose > 5:
              print(f"  Skipping protocol {row[1]} for {row[0]}")
    # List of requested protocols (target names)
    requested_protocols = []
    for pm in sel_js["protocol_map"]:
      if pm[1] not in requested_protocols:
        requested_protocols.append(pm[1])

    # Extract slices for findings from dicoms
    samples = {}
    for p in patient:
      if args.verbose > 0:
        print(f"{p}:")
      for f in patient[p]:
        found_slices = {}
        for sname in slices[p][str(f)]:
          if args.verbose > 2:
            print(f"    FID {f} - {sname}: {slices[p][str(f)][sname]}")
          # Find the stack folder
          path = os.path.join(args.data, group, p)
          # Assume each patient has one directory with the scans (we use the first one found and warn if there are more)
          scan_dir = []
          for dir in os.listdir(path):
            if os.path.isdir(os.path.join(path,dir)):
              scan_dir.append(dir)
          if len(scan_dir) == 0:
            raise Exception(f"Patient {p} has no folder with scans")
          elif len(scan_dir) > 1:
            timestamps = [(l,"-".join(reversed(d.split("-")[0:3]))) for l,d in enumerate(scan_dir)]
            timestamps = sorted(timestamps, key=lambda d: tuple(map(int, d[1].split('-'))))
            scan_dir = scan_dir[timestamps[0][0]]
            print(f"Warning: multiple folders for patient {p}; using {scan_dir}")
          else:
            scan_dir = scan_dir[0]

          found = False
          for pm in sel_js['protocol_map']:
            if sname == pm[0]:
              folder_ser_num = ser_num[p][pm[1]][f]
              found = True
          if not found:
            raise Exception(f"Protrocol {sname} not found in protocol_map")
          path = os.path.join(path,scan_dir)
          folder =None
          folder_num = None
          for dir in os.listdir(path):
            prot = "".join(dir.split("-")[1:]).replace(" ","").split(".")[0].rstrip("01234567890")
            if prot == sname:
              num = int(dir.split("-")[0])
              if folder is not None:
                if prot == 't2tsetra' or prot == 'ep2ddifftraDYNDISTADC' or \
                   prot == 'ep2ddifftraDYNDISTCALCBVAL' or prot == 'ep2ddifftra2x2Noise0FSDYNDISTADC' or \
                   prot == 'ep2ddifftra2x2Noise0FSDYNDISTCALCBVAL': # if more folders, use ser_num
                  if num == folder_ser_num:
                    folder = os.path.join(path,dir)
                    folder_num = num
                else: # Unknown protocol for which there is more than one folder
                  raise Exception(f"Found two stacks for protocol {prot}/{sname}: {folder}, {os.path.join(path,dir)}")
              else:
                folder = os.path.join(path,dir)
                folder_num = num
          if folder is not None and folder_num != folder_ser_num:
            raise Exception(f"For {p} we found dicom serial {folder_num}, but requested {folder_ser_num}")
          if folder is None:
            print(f"Warning: {sname} stack for patient {p} not found")
          else:
            # Find slice file
            # - while files may not be in order of stack, the numbering is concescutive and the
            #   counting starts from 0, so a slice number requested for which there is not file
            #   should mean the slice is not in the data, whatever the order.
            slice_file = os.path.join(folder,f"{slices[p][str(f)][sname][-1]:06d}.dcm")
            if os.path.isfile(slice_file):
              # Not all slices specified seem to exist
              found = False
              for pm in sel_js['protocol_map']:
                if sname == pm[0]:
                  found = True
                  if ser_num[p][pm[1]][f] != folder_num:
                    raise Exception(f"Stack number {folder_num} does not match dicom serial {ser_num[p][pm[1]][f]} for {p}, {pm[1]}, finding {f}")
                  found_slices[pm[1]] = (slice_file, slices[p][str(f)][sname], ser_num[p][pm[1]])
              if not found:
                raise Exception(f"Did not find slice for {sname}")
            else:
              print(f"Warning: slice {slice_file} is missing")
        # Check if finding has all protocols
        missing = []
        for rs in requested_protocols:
          if rs not in found_slices:
            if rs == "dwi_c":
              if rs not in [f[0:5] for f in found_slices]:
                missing.append(rs)
            else:
              missing.append(rs)
        if len(missing) > 0:
          if args.verbose > 0:
            print(f"  FID {f} - incomplete! Missing: {', '.join([m for m in missing])}")
        else:
          if args.verbose > 0:
            print(f"  FID {f} - " + (', '.join(found+"-"+(str(found_slices[found][2][f])+str(found_slices[found][1])) for found in found_slices.keys())))
          if p not in samples:
            samples[p] = {}
          samples[p][f] = found_slices

    # Convert samples: samples[PATIENT][FID][PROTOCOL] = [FILE, FINDING_LOCATION, NAME]
    if args.verbose > 0:
      print("# Creating dataset")
    ref_protocol = sel_js["ref_protocol"]
    mask_d = sel_js["mask_size"]/2
    if args.disable_parallel:
      for patient in samples:
        prostatex_slices(patient, samples[patient], ref_protocol, world_matrix[patient],
                         pos[patient], label[patient], group, mask_d,
                         sel_js['width'], sel_js['height'], sel_js['density'],
                         args.out, args.force, args.check, args.verbose)
    else:
      import joblib
      d_inp = joblib.Parallel(n_jobs=-1, prefer="processes", verbose=10*args.verbose) \
                (joblib.delayed(prostatex_slices)(patient, samples[patient], ref_protocol, world_matrix[patient],
                                                  pos[patient], label[patient], group, mask_d,
                                                  sel_js['width'], sel_js['height'], sel_js['density'],
                                                  args.out, args.force, args.check, -1)
                  for patient in samples)

def prostatex_slices(patient, samples_patient, ref_protocol, world_matrix_patient,
                     pos_patient, label_patient, group, mask_d, width, height, density,
                     out, force, check, verbose=0):
  if verbose == -1: # parallel
    print(f"Starting {patient}")
  used_findings = []
  for finding in samples_patient:
    if finding not in used_findings:
      if verbose > 0:
        print(f"  {patient} - {finding}")
      if ref_protocol not in samples_patient[finding]:
        raise Exception(f"Ref. protocol not available for {patient}, {finding}")
      dicom0, info0 = read_dicom(samples_patient[finding][ref_protocol][0])
      ref_transf_slice2patient = world_matrix_patient[ref_protocol][finding].copy()
      # Correct reference slice transformation, relative to slice, not stack
      ref_slice_num = int(os.path.basename(samples_patient[finding][ref_protocol][0]).split(".")[0])
      slice_pos_correction = ref_transf_slice2patient @ np.array([0.0, 0.0, ref_slice_num, 0.0])
      ref_transf_slice2patient[0:3,3] += np.array(slice_pos_correction[0:3])
      # Ref. size and scan number
      ref_size = dicom0.pixel_array.shape
      ref_slice_scan = "%02d" % \
          int(os.path.basename(os.path.dirname(samples_patient[finding][ref_protocol][0])).split("-")[0])
      # Create Slices
      for protocol in samples_patient[finding]:
        if verbose > 0:
          print(f"    {protocol} - {os.path.basename(os.path.dirname(samples_patient[finding][protocol][0]))}")
        protocol_ext = ""
        # Load stack for protocol
        data = []
        info = None
        for path in glob.glob(os.path.join(os.path.dirname(samples_patient[finding][protocol][0]),
                                           "*.dcm")):
          dicom, info0 = read_dicom(path)
          if info is None:
            info = info0
            if protocol[0:5] == "dwi_c":
              protocol_ext = "-"+str(int(info['[B_value]']))
          data.append(dicom)
        data = sorted(data, key=lambda d: d.SliceLocation)
        data = np.stack([d.pixel_array.astype(np.uint16, order='C', casting='safe', copy=False) for d in data])
        # Save slice data
        pid = patient.replace("-","")
        stack_scan = "%02d" % \
            int(os.path.basename(os.path.dirname(samples_patient[finding][protocol][0])).split("-")[0])
        create_slice(os.path.join(out,pid), data, 0, stack_scan,
                     info, world_matrix_patient[protocol][finding], protocol, protocol_ext,
                     pid, group, ref_slice_scan, width, height,
                     ref_slice_scan, ref_slice_num,
                     ref_transf_slice2patient, ref_size,
                     density, force, check, verbose)
      # Create masks
      x_scale = np.float64(width)/np.float64(ref_size[1])
      y_scale = np.float64(height)/np.float64(ref_size[0])
      transf_slice = np.matrix([[x_scale,0.0,0.0,0.0],
                                [0.0,y_scale,0.0,0.0],
                                [0.0,0.0,1.0,0.0],
                                [0.0,0.0,0.0,1.0]],dtype=np.float64)
      rois = {}
      for finding in samples_patient:
        if finding not in used_findings:
          pp = np.linalg.inv(world_matrix_patient[ref_protocol][finding]) @ pos_patient[ref_protocol][finding]
          pp = pp[0:3]/pp[3]
          if np.floor(pp[2]+0.5) == ref_slice_num:
            used_findings.append(finding) # Mark those findings used to avoid replicating slices
            tag = label_patient[str(finding)]
            if tag not in rois:
              rois[tag] = []
            rois[tag].append({
                "x": [pp[0]-mask_d, pp[0]+mask_d, pp[0]+mask_d, pp[0]-mask_d],
                "y": [pp[1]+mask_d, pp[1]+mask_d, pp[1]-mask_d, pp[1]-mask_d]
              })
      for tag in rois:
        if verbose > 0:
          print(f"    {tag} mask")
        mask_slice = np.zeros((height,width))
        for p in rois[tag]:
          draw_roi(p['x'],p['y'],mask_slice, transf_slice)
        if np.sum(mask_slice) > 0: # Only store if we have any region at all
          save_slice(os.path.join(out,pid), f"{pid}-{group}-{ref_slice_scan}-{ref_slice_num:04d}-{tag}",
                     mask_slice, force, check, verbose)
      # Masks for ProstateX Masks repo
      import nibabel as  nib
      dir = os.path.dirname(samples_patient[finding][ref_protocol][0])
      while not os.path.isdir(os.path.join(dir,"PROSTATEx_masks")):
        dir = os.path.dirname(dir)
        if len(dir) < 2:
          raise Exception("PROSTATEx_masks not found")
      dir = os.path.join(dir,"PROSTATEx_masks","Files")
      if ref_protocol == "t2-tra":
        prot_dir = "T2"
      elif ref_protocol == "adc":
        prot_dir = "ADC"
      else:
        print(f"Warning: no ProstateX masks for reference protocol {ref_protocol}")
        return
      # Lesion ROIs for all findings
      rois = {}
      used_findings = []
      for finding in samples_patient:
        if finding not in used_findings:
          pp = np.linalg.inv(world_matrix_patient[ref_protocol][finding]) @ pos_patient[ref_protocol][finding]
          if np.floor(pp[2]+0.5) == ref_slice_num:
            used_findings.append(finding) # Mark those findings used to avoid replicating slices
            tag = label_patient[str(finding)].replace("_sq","")
            if tag not in rois:
              rois[tag] = []
            pattern = "ProstateX-"+pid[-4:]+"-Finding"+str(finding)+"-*_ROI.nii.gz"
            path = os.path.join(dir,"lesions","Masks",prot_dir,pattern)
            mask_files = glob.glob(path)
            if len(mask_files) == 0:
              print(f"Warning, ROI for {pattern} not found, skipping")
            elif len(mask_files) != 1:
              raise Exception(f"Too many ROIs for {pattern}")
            else:
              mask = nib.load(mask_files[0]).get_fdata(dtype=np.float64)
              # Row/column vs. x/y exchanges (y is row!); origin along at bottom/top of y axis
              rois[tag] = np.flip(np.transpose(mask[:,:,ref_slice_num]), axis=0)
              if rois[tag].shape != (height,width):
                xs = rois[tag].shape[1] / width
                ys = rois[tag].shape[0] / height
                rois[tag] = ndi.affine_transform(rois[tag],
                                                 np.array([[ys, 0],
                                                           [0, xs]]),
                                                 output_shape=(height,width))
              mi = np.amin(rois[tag])
              ma = np.amax(rois[tag])
              th = 0.5 if np.abs(ma-mi) < 1e-4 else (mi + ma)/2.0
              rois[tag][rois[tag] < th] = 0.0
              rois[tag][rois[tag] > 0.0] = 1.0
      for tag in rois:
        if verbose > 0:
          print(f"    {tag} mask")
        if np.sum(rois[tag]) > 0: # Only store if we have any region at all
          save_slice(os.path.join(out,pid), f"{pid}-{group}-{ref_slice_scan}-{ref_slice_num:04d}-{tag}",
                     rois[tag], force, check, verbose)

      # Anatomical segmentation masks
      rois = {}
      for tag in ['prostate', 'pz', 'tz']:
        if tag not in rois:
          rois[tag] = []
        if tag == "prostate":
          fname = "ProstateX-"+pid[-4:]+".nii.gz"
        else:
          fname = "ProstateX-"+pid[-4:]+"_"+tag+".nii.gz"
        mask_file = os.path.join(dir,"prostate","mask_"+tag,fname)
        if not os.path.isfile(mask_file):
          if tag == "prostate":
            fname = "ProstateX-"+pid[-3:]+".nii.gz"
          else:
            fname = "ProstateX-"+pid[-3:]+"_"+tag+".nii.gz"
          mask_file = os.path.join(dir,"prostate","mask_"+tag,fname)
        if not os.path.isfile(mask_file):
          if tag == "prostate":
            fname = "VOLUME-"+pid[-4:]+".nii.gz"
          else:
            fname = "VOLUME-"+pid[-4:]+"_"+tag+".nii.gz"
          mask_file = os.path.join(dir,"prostate","mask_"+tag,fname)
        if not os.path.isfile(mask_file):
          print(f"Warning, {tag} mask for {fname} not found, skipping")
        else:
          mask = nib.load(mask_file).get_fdata(dtype=np.float64)
          rois[tag] = np.flip(np.transpose(mask[:,:,ref_slice_num]), axis=0)
          if rois[tag].shape != (height,width):
            xs = rois[tag].shape[1] / width
            ys = rois[tag].shape[0] / height
            rois[tag] = ndi.affine_transform(rois[tag],
                                             np.array([[ys, 0],
                                                       [0, xs]]),
                                             output_shape=(height,width))
          mi = np.amin(rois[tag])
          ma = np.amax(rois[tag])
          th = 0.5 if np.abs(ma-mi) < 1e-4 else (mi + ma)/2.0
          rois[tag][rois[tag] < th] = 0.0
          rois[tag][rois[tag] > 0.0] = 1.0
          if tag == "prostate": # Fill holes for these tags
            rois[tag] = ndi.binary_fill_holes(rois[tag]).astype(np.float64)
      for tag in rois:
        if verbose > 0:
          print(f"    {tag} mask")
        if np.sum(rois[tag]) > 0: # Only store if we have any region at all
          save_slice(os.path.join(out,pid), f"{pid}-{group}-{ref_slice_scan}-{ref_slice_num:04d}-{tag}",
                     rois[tag], force, check, verbose)

  if verbose == -1: # parallel
    print(f"Stopping {patient}")

def save_slice(outdir, fn, data, force, check, verbose):
  # Save result
  if not os.path.isdir(outdir):
    os.makedirs(outdir)
  check_overwrite = False
  fn_base = os.path.join(outdir,fn)
  if check and os.path.isfile(fn_base+".npy"):
    old_data = np.load(fn_base+".npy")
    try:
      diff = np.linalg.norm(np.subtract(data, old_data), ord=1)
      if diff > 1e-6:
        print(f"Warning: {fn_base} data differs by {diff}")
        if force == True:
          check_overwrite = True
    except Exception as e:
      print(f"Warning: {fn_base} data is not comparable:\n{e}")
  update_image = False
  if (force and not check) or check_overwrite or not os.path.isfile(fn_base+".npy"):
    if verbose > 0:
      print("    Saving npy")
    np.save(fn_base+".npy", data, allow_pickle=False)
    update_image = True
  if (force and not check) or update_image or not os.path.isfile(fn_base+".png"):
    # For display only - we scale each image from its min to max value to full intensity range
    if verbose > 0:
      print("    Saving png")
    dmax = data.max()
    dmin = data.min()
    if dmax > dmin:
      data = ((data - dmin) / (dmax - dmin))
    data *= (2**16-1)
    Image.fromarray(data.astype(np.uint16)).save(fn_base+".png", optimize=True, bits=16)

if __name__ == '__main__':
  main()
