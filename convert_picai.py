#!/usr/bin/env python3
#
# convert_picai.py - QDicom Utilities
# Create dataset for deep learning from PI-CAI data.
#
# SPDX-FileCopyrightText: Copyright (C) 2023 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Usage example:
#
# ./convert_picai.py -d data-picai/images -l data-picai/picai_labels -o ds-picai01 -v
#
# See --help for full arguments
#
# This converts the full dataset into our numpy-based representation:
#
#  dataset/PATIENT_ID-SESSION_ID/PATIENT_ID-SESSION_ID-SCAN_ID-SLICE-PROTOCOL.{npy,png}
#
# It requires the PI-CAI dataset and the labels, in separate folds given via the arguments.

import os
import argparse
import glob
import json
import math
import numpy as np
import sobol_seq
import SimpleITK as sitk
import nibabel as nib
import nibabel.processing as nibp
import tempfile
import csv
from PIL import Image, ImageDraw
from skimage.draw import polygon, line
import scipy.ndimage as ndi

from read_dicom_siemens import read_dicom

def main():
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', type=str, help='Folder with the dicom files (structure: foldX/PATIENT/DATA.mha')
  parser.add_argument('-l', '--labels', type=str, help='Folder with the label files for PI-CAI')
  parser.add_argument('-o', '--out', type=str, help='Folder to store dataset (stores as REF_PATIENT-REF_SESSION/REF_PATIENT-REF_SESSION-REF_SCAN-REF_SLICE-PROTOCOL in npy; png only for display)')
  parser.add_argument('-a', '--all-slices', action='store_true', help='Convert all slices, not only prostate slices')
  parser.add_argument('-m', '--modalities', type=lambda x : x.lower(), nargs='+',
                      default=sorted(['t2w', 'adc', 'hbv', 'prostate', 'pirads']),
                      choices=['t2w', 'adc', 'hbv', 'cor', 'sag', 'prostate', 'pirads', 'suspicious', 'normal'],
                      help='Select modalities and masks to convert from t2w, adc, hbv, cor, sag, prostate, pirads, suspicious, normal; make sure t2w is always selected as it is reference frame')
  parser.add_argument('-p', '--disable-parallel', action='store_true', help='Do not execute in parallel (disable for testing, etc)')
  parser.add_argument('-v', '--verbose', action='count', help='Increase output verbosity', default=0)
  args = parser.parse_args()

  # Check arguments
  if args.data == None or not os.path.isdir(args.data):
    print(f"Dicom folder not specified or is not a directory: {args.data}")
    exit(1)
  if args.labels == None or not os.path.isdir(args.labels):
    print(f"Labels folder not specified or is not a directory: {args.labels}")
    exit(1)
  if args.out == None:
    print("Dataset folder not specified")
    exit(1)

  conv_picai(args.data, args.labels, args.out, args.all_slices, args.modalities, 
             args.verbose, not(args.disable_parallel))

def conv_picai(data, labels, out, all_slices, modalities, verbose=0, parallel=True):
  # Convert all patient data
  patients = []
  for fold in sorted(os.listdir(data)): # for each fold
    if os.path.isdir(os.path.join(data,fold)):
      for p in sorted(os.listdir(os.path.join(data, fold))): # for each patient
        patients.append((fold,p))

  # Read csv data
  patient_info = {}
  with open(os.path.join(labels,'clinical_information', 'marksheet.csv'), mode='r') as csvfile:
    csv_reader = csv.DictReader(csvfile, delimiter=',')
    for row in csv_reader:
      pid = row['patient_id']+"-"+row['study_id']
      patient_info[pid] = { }
      for tag in row:
        patient_info[pid][tag] = row[tag]

  if parallel:
    import joblib
    joblib.Parallel(n_jobs=-1, prefer="processes", verbose=10*verbose) \
                   (joblib.delayed(process_patient)(data, labels, out, ps, patient_info,
                                   all_slices, modalities, verbose) for ps in patients)
  else:
    for ps in patients:
      process_patient(data, labels, out, ps, patient_info, all_slices, modalities, verbose)

def process_patient(data, labels, out, ps, pinfo, all_slices, modalities, verbose=0):
  # Convert patient
  fold = ps[0]
  p = ps[1]
    
  # Load patient stacks as nifti files
  with tempfile.TemporaryDirectory() as tmp: # Temporary folder to store nii conversions
                                             # FIXME: can this be done in memory?

    # Handle more than one study in folder and split into two studies/sessions
    sessions = []
    for f in sorted(os.listdir(os.path.join(data, fold, p))):
      study_id = os.path.splitext(f)[0].split("_")[1]
      if study_id not in sessions:
        sessions.append(study_id)

    # Process each session separately and count (for our session ids)
    for session_num, session in enumerate(sessions):

      # Get stacks for the patient session
      stacks = {}
      for f in sorted(os.listdir(os.path.join(data, fold, p))):
        study_id = os.path.splitext(f)[0].split("_")[1]
        if study_id == session: # skip other session files
          proto = os.path.splitext(f)[0].split("_")[-1]
          if proto in modalities: # Convert only selected
            if verbose > 0:
              print(f"# Patient {fold}/{p} ({session}) -- {proto}")
            # Convert MHA to NII temporarily to load/process with nib
            img = sitk.ReadImage(os.path.join(data, fold, p, f))
            fn = os.path.join(tmp, f+'.nii')
            sitk.WriteImage(img, fn)
            stacks[proto] = nib.load(fn)
      if "t2w" not in stacks:
        raise Exception(f"No T2W reference stack for {fold}/{p} ({session})")

      # Target size to write out is taken from t2w info
      target_dim = stacks["t2w"].get_fdata().shape # Size from t2w
      outdir = os.path.join(out,f"{p}-{session_num+1:02d}")
      min_slice, max_slice = get_prostate_slices(labels, p, session, all_slices)
      if min_slice is None:
        if verbose > 0:
          print(f"{p}-{session}: no prostate delineation")
        continue # Skip patient/session as no prostate delineation
      if not os.path.isdir(outdir):
        os.makedirs(outdir)

      # Store original patient number and session number in json to link it to PI-CAI data
      info = pinfo[p+"-"+session]
      info['fold'] = fold
      info['session'] =f"{session_num+1:02d}"
      with open(os.path.join(outdir,f"{p}-{session_num+1:02d}.json"),'w') as fo:
        fo.write(json.dumps(info, indent=2, sort_keys=True))

      # Transform and write to numpy/png stacks
      if verbose > 0:
        print(f"# Patient {fold}/{p} ({session}) -- convert slices {min_slice}-{max_slice-1}")
      for num, proto in enumerate(stacks):
        dim = stacks[proto].header.get_data_shape()
        vs = stacks[proto].header.get_zooms() # Scaling/zooms from stack info
        sx = dim[0] * vs[0] / target_dim[0]
        sy = dim[1] * vs[1] / target_dim[1]
        sz = dim[2] * vs[2] / target_dim[2]
        # Transform stack to target orientation and resolution (t2w stack)
        if proto == "t2w": # Reference frame, so no need to convert
          X = stacks[proto].get_fdata().astype(np.float32)
        else:
          transf_stack = nibp.conform(stacks[proto], out_shape=target_dim, voxel_size=(sx,sy,sz), orientation='LPS')
          X = transf_stack.get_fdata().astype(np.float32)
        # Save transformed stack as numpy and png
        fn_base = os.path.join(outdir,f"{p}-{session_num+1:02d}-{num+1:04d}")
        if proto == "hbv":
          proto_str = "dwi_c-1000" # computed dwi >= 1000 (also called high-b-value, but for consistency with other data renamed)
        elif proto == "cor":
          proto_str = "t2-cor" # rename for consistency
        elif proto == "sag":
          proto_str = "t2-sag" # rename for consistency
        elif proto == "t2w":
          proto_str = "t2-tra" # rename for consistency
        else:
          proto_str = proto
        for slice in range(min_slice,max_slice):
          fn = fn_base + f"-{slice:04d}-{proto_str}"
          XX = np.copy(X[:,:,slice])
          np.save(fn+".npy", XX, allow_pickle=False)
          dmax = XX.max()
          dmin = XX.min()
          if dmax > dmin:
            XX = ((XX - dmin) / (dmax - dmin))
          XX *= (2**16-1)
          Image.fromarray(XX.astype(np.uint16)).save(fn+".png", optimize=True, bits=16)

      # Process delineations for patient / session
      if verbose > 0:
        print(f"# Patient {fold}/{p} ({session}) -- delineations")

      # Resampled, human expert csPCa lesion delineations resampled to t2w (so no transf. needed)
      path = os.path.join(labels, "csPCa_lesion_delineations", "human_expert", "resampled", f"{p}_{session}.nii.gz")
      if os.path.isfile(path): # If not we do not have data (does not mean negative, but missing data, see PI-CAI doc)
        csPCa = nib.load(path)
        X = csPCa.get_fdata().astype(np.uint8)
        if verbose > 0:
          print(f"# Patient {fold}/{p} ({session}) -- PIRADS: {np.unique(X)}")
        for slice in range(min_slice,max_slice):
          if 'pirads' in modalities:
            # PI-RADS rating
            fn = os.path.join(outdir,f"{p}-{session_num+1:02d}-9999-{slice:04d}-pirads")
            XX = np.copy(X[:,:,slice])
            np.save(fn+".npy", XX, allow_pickle=False)
            # PIRADS PNG
            dmax = XX.max()
            dmin = XX.min()
            if dmax > dmin:
              XX = ((XX - dmin) / (dmax - dmin))
            XX *= (2**8-1)
            Image.fromarray(XX.astype(np.uint8)).save(fn+".png", optimize=True, bits=8)
          if 'suspicious' in modalities:
            # cs-PCa as suspicious map (it's cs-PCa if PIRADS is 3,4,5; see PI-CAI doc)
            fns = os.path.join(outdir,f"{p}-{session_num+1:02d}-9998-{slice:04d}-suspicious")
            XX = np.copy(X[:,:,slice])
            XX[XX<3] = 0
            XX[XX>2] = 1
            np.save(fns+".npy", XX, allow_pickle=False)
            # Suspicious PNG
            dmax = XX.max()
            dmin = XX.min()
            if dmax > dmin:
              XX = ((XX - dmin) / (dmax - dmin))
            XX *= (2**8-1)
            Image.fromarray(XX.astype(np.uint8)).save(fns+".png", optimize=True, bits=8)
          if 'normal' in modalities:
            # cs-PCa as normal map (it's cs-PCa if PIRADS is 1,2; see PI-CAI doc)
            fns = os.path.join(outdir,f"{p}-{session_num+1:02d}-9997-{slice:04d}-normal")
            XX = np.copy(X[:,:,slice])
            XX[XX>2] = 0
            XX[XX>0] = 1
            np.save(fns+".npy", XX, allow_pickle=False)
            # Suspicious PNG
            dmax = XX.max()
            dmin = XX.min()
            if dmax > dmin:
              XX = ((XX - dmin) / (dmax - dmin))
            XX *= (2**8-1)
            Image.fromarray(XX.astype(np.uint8)).save(fns+".png", optimize=True, bits=8)

      # Anatomical AI delineation for whole gland, for t2w (so no transf. needed)
      if 'prostate' in modalities:
        path = os.path.join(labels, "anatomical_delineations", "whole_gland", "AI", "Bosma22b", f"{p}_{session}.nii.gz")
        if os.path.isfile(path): # if not, means we have no data
          prostate = nib.load(path)
          X = prostate.get_fdata().astype(np.uint8)
          fn_base = os.path.join(outdir,f"{p}-{session_num+1:02d}-9900")
          for slice in range(min_slice,max_slice):
            # Prostate mask
            fn = fn_base + f"-{slice:04d}-prostate"
            XX = np.copy(X[:,:,slice])
            np.save(fn+".npy", XX, allow_pickle=False)
            # Prostate png
            dmax = XX.max()
            dmin = XX.min()
            if dmax > dmin:
              XX = ((XX - dmin) / (dmax - dmin))
            XX *= (2**8-1)
            Image.fromarray(XX.astype(np.uint8)).save(fn+".png", optimize=True, bits=8)

def get_prostate_slices(labels, p, session, all_slices):
  # Determine slice range of prostate for patient p/session; if all_slices report full slice range
  path = os.path.join(labels, "anatomical_delineations", "whole_gland", "AI", "Bosma22b", f"{p}_{session}.nii.gz")
  if os.path.isfile(path): # if not, means we have no data
    prostate = nib.load(path)
    X = prostate.get_fdata().astype(np.uint8)
    if all_slices:
      return 0, X.shape[-1]
    min_slice = X.shape[-1]
    max_slice = 0
    for slice in range(0,X.shape[-1]):
      cnt = np.unique(X[:,:,slice]).shape[0]
      if cnt > 1:
        if slice < min_slice:
          min_slice = slice
        if slice > max_slice:
          max_slice = slice
    return min_slice, max_slice+1
  return None, None

if __name__ == '__main__':
  main()
