#!/usr/bin/env python3
#
# convert_picai.py - QDicom Utilities
# Create dataset for deep learning from PI-CAI data.
#
# SPDX-FileCopyrightText: Copyright (C) 2023, 2026 Frank C Langbein <frank@langbein.org>, Cardiff University
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
import json
import numpy as np
import SimpleITK as sitk
import nibabel as nib
import nibabel.processing as nibp
import tempfile
import csv
from PIL import Image

def _slice_npy_needs_write(npy_path, arr):
  """Return True if .npy is missing or its array differs from arr."""
  arr = np.asarray(arr)
  if not os.path.isfile(npy_path):
    return True
  try:
    existing = np.load(npy_path, mmap_mode='r')
    try:
      if existing.shape != arr.shape or existing.dtype != arr.dtype:
        return True
      # Integer masks must match exactly; float slices may differ slightly after resampling.
      if np.issubdtype(arr.dtype, np.floating):
        return not np.allclose(existing, arr, rtol=1e-5, atol=1e-6, equal_nan=True)
      return not np.array_equal(existing, arr)
    finally:
      del existing
  except Exception:
    return True

def save_npy_and_png_if_changed(fn_base, arr, png_bit_depth):
  """Write fn_base.npy and fn_base.png only when .npy content would change.

  Float arrays are compared with allclose (see _slice_npy_needs_write); integer masks use exact equality.

  Args:
      fn_base (str): Path without .npy / .png suffix.
      arr (ndarray): 2D slice array to store.
      png_bit_depth (int): 8 for masks, 16 for modality previews.
  """
  npy_path = fn_base + ".npy"
  png_path = fn_base + ".png"
  arr = np.asarray(arr)
  if not _slice_npy_needs_write(npy_path, arr):
    return
  np.save(npy_path, arr, allow_pickle=False)
  XX = arr.astype(np.float32, copy=True)
  dmax = XX.max()
  dmin = XX.min()
  if dmax > dmin:
    XX = (XX - dmin) / (dmax - dmin)
  if png_bit_depth == 16:
    XX *= (2**16 - 1)
    Image.fromarray(XX.astype(np.uint16)).save(png_path, optimize=True, bits=16)
  else:
    XX *= (2**8 - 1)
    Image.fromarray(XX.astype(np.uint8)).save(png_path, optimize=True, bits=8)

def main():
  """Main function to parse arguments and convert PI-CAI data.

  Parses command line arguments and calls conv_picai to convert
  PI-CAI DICOM data into a numpy-based representation.
  """
  # Parse arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--data', type=str, help='Folder with the dicom files (structure: foldX/PATIENT/DATA.mha')
  parser.add_argument('-l', '--labels', type=str, help='Folder with the label files for PI-CAI')
  parser.add_argument('-o', '--out', type=str, help='Folder to store dataset (stores as REF_PATIENT-REF_SESSION/REF_PATIENT-REF_SESSION-REF_SCAN-REF_SLICE-PROTOCOL in npy; png only for display)')
  parser.add_argument('-a', '--all-slices', action='store_true', help='Convert all slices, not only prostate slices')
  parser.add_argument('-m', '--modalities', type=lambda x : x.lower(), nargs='+',
                      default=sorted(['t2w', 'adc', 'hbv', 'prostate', 'pztz', 'pirads']),
                      choices=['t2w', 'adc', 'hbv', 'cor', 'sag', 'prostate', 'pztz', 'pirads', 'suspicious', 'normal'],
                      help='Select modalities and masks to convert from t2w, adc, hbv, cor, sag, prostate, pirads, suspicious, normal; make sure t2w is always selected as it is reference frame')
  parser.add_argument('-p', '--disable-parallel', action='store_true', help='Do not execute in parallel (disable for testing, etc)')
  parser.add_argument('-v', '--verbose', action='count', help='Increase output verbosity', default=0)
  args = parser.parse_args()

  # Check arguments
  if args.data is None or not os.path.isdir(args.data):
    print(f"Dicom folder not specified or is not a directory: {args.data}")
    exit(1)
  if args.labels is None or not os.path.isdir(args.labels):
    print(f"Labels folder not specified or is not a directory: {args.labels}")
    exit(1)
  if args.out is None:
    print("Dataset folder not specified")
    exit(1)

  conv_picai(args.data, args.labels, args.out, args.all_slices, args.modalities,
             args.verbose, not(args.disable_parallel))

def conv_picai(data, labels, out, all_slices, modalities, verbose=0, parallel=True):
  """Convert PI-CAI data to numpy-based representation.

  Args:
      data (str): Path to folder with DICOM files
      labels (str): Path to folder with label files
      out (str): Output folder for dataset
      all_slices (bool): Convert all slices, not only prostate slices
      modalities (list): List of modalities to convert
      verbose (int, optional): Verbosity level. Defaults to 0
      parallel (bool, optional): Execute in parallel. Defaults to True
  """
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
  """Process a single patient's data.

  Args:
      data (str): Path to DICOM data folder
      labels (str): Path to labels folder
      out (str): Output folder
      ps (tuple): Patient information (fold, patient_id)
      pinfo (dict): Patient information dictionary
      all_slices (bool): Convert all slices or only prostate slices
      modalities (list): List of modalities to convert
      verbose (int, optional): Verbosity level. Defaults to 0

  Raises:
      Exception: If no T2W reference stack is found
  """
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
      pinfo_key = p + "-" + session
      if pinfo_key not in pinfo:
        if verbose > 0:
          print(f"{p}-{session}: no clinical_information row ({pinfo_key})")
        continue
      if not os.path.isdir(outdir):
        os.makedirs(outdir)

      # Store original patient number and session number in json to link it to PI-CAI data
      info = pinfo[pinfo_key]
      info['fold'] = fold
      info['session'] =f"{session_num+1:02d}"
      with open(os.path.join(outdir,f"{p}-{session_num+1:02d}.json"),'w') as fo:
        fo.write(json.dumps(info, indent=2, sort_keys=True))

      # Transform and write to numpy/png stacks
      if verbose > 0:
        print(f"# Patient {fold}/{p} ({session}) -- convert slices {min_slice}-{max_slice-1}")
      for num, proto in enumerate(stacks):
        # Transform stack to target orientation and resolution (t2w stack)
        if proto == "t2w": # Reference frame, so no need to convert
          X = stacks[proto].get_fdata().astype(np.float32)
        else:
          dim = stacks[proto].header.get_data_shape()
          vs = stacks[proto].header.get_zooms() # Scaling/zooms from stack info
          sx = dim[0] * vs[0] / target_dim[0]
          sy = dim[1] * vs[1] / target_dim[1]
          sz = dim[2] * vs[2] / target_dim[2]
          transf_stack = nibp.conform(stacks[proto], out_shape=target_dim, voxel_size=(sx,sy,sz), orientation='LPS')
          X = transf_stack.get_fdata().astype(np.float32)
        # Save transformed stack as numpy and png
        fn_base = os.path.join(outdir,f"{p}-{session_num+1:02d}-0001") # Use 0001 as reference slice number (assumed all aligned)
        if proto == "hbv":
          proto_str = "dwi_c-1000" # computed dwi >= 1000 (also called high-b-value, but for consistency with other datasets we rename)
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
          save_npy_and_png_if_changed(fn, X[:,:,slice], 16)

      # Process delineations for patient / session

      # Human expert csPCa lesion delineations on the T2W grid (resampled first; Pooch25 fills gaps where resampled is absent).
      cspca_dir = os.path.join(labels, "csPCa_lesion_delineations", "human_expert")
      path_resampled = os.path.join(cspca_dir, "resampled", f"{p}_{session}.nii.gz")
      path_pooch25 = os.path.join(cspca_dir, "Pooch25", f"{p}_{session}.nii.gz")
      if os.path.isfile(path_resampled):
        cspca_path = path_resampled
        cspca_source = "resampled"
      elif os.path.isfile(path_pooch25):
        cspca_path = path_pooch25
        cspca_source = "Pooch25"
      else:
        cspca_path = None
        cspca_source = None
      if cspca_path is not None:
        if verbose > 0:
          print(f"# Patient {fold}/{p} ({session}) -- PCa labels ({cspca_source})")
        csPCa = nib.load(cspca_path)
        X = csPCa.get_fdata().astype(np.uint8)
        if verbose > 0:
          print(f"# Patient {fold}/{p} ({session}) -- PIRADS: {np.unique(X)}")
        for slice in range(min_slice,max_slice):
          if 'pirads' in modalities:
            # PI-RADS rating
            fn = os.path.join(outdir,f"{p}-{session_num+1:02d}-0001-{slice:04d}-pirads")
            save_npy_and_png_if_changed(fn, X[:,:,slice], 8)
          if 'suspicious' in modalities:
            # cs-PCa as suspicious map (it's cs-PCa if PIRADS is 3,4,5; see PI-CAI doc)
            fns = os.path.join(outdir,f"{p}-{session_num+1:02d}-0001-{slice:04d}-suspicious")
            XX = np.copy(X[:,:,slice])
            XX[XX<3] = 0
            XX[XX>2] = 1
            save_npy_and_png_if_changed(fns, XX, 8)
          if 'normal' in modalities:
            # cs-PCa as normal map (it's cs-PCa if PIRADS is 1,2; see PI-CAI doc)
            fns = os.path.join(outdir,f"{p}-{session_num+1:02d}-0001-{slice:04d}-normal")
            XX = np.copy(X[:,:,slice])
            XX[XX>2] = 0
            XX[XX>0] = 1
            save_npy_and_png_if_changed(fns, XX, 8)
      elif verbose > 0 and ('pirads' in modalities or 'suspicious' in modalities or 'normal' in modalities):
        print(f"# Patient {fold}/{p} ({session}) -- no csPCa delineation (resampled/Pooch25)")

      # Anatomical AI delineation for whole gland, for t2w (so no transf. needed)
      if 'prostate' in modalities:
        for src in ['Bosma22b', 'Guerbet23']:
          if verbose > 0:
            print(f"# Patient {fold}/{p} ({session}) -- prostate labels {src}")
          path = os.path.join(labels, "anatomical_delineations", "whole_gland", "AI", src, f"{p}_{session}.nii.gz")
          if os.path.isfile(path): # if not, means we have no data
            prostate = nib.load(path)
            X = prostate.get_fdata().astype(np.uint8)
            fn_base = os.path.join(outdir,f"{p}-{session_num+1:02d}-0001")
            for slice in range(min_slice,max_slice):
              # Prostate mask
              fn = fn_base + f"-{slice:04d}-prostate:{src}"
              save_npy_and_png_if_changed(fn, X[:,:,slice], 8)
      if 'pztz' in modalities:
        for src in ['HeviAI23', 'Yuan23']:
          if verbose > 0:
            print(f"# Patient {fold}/{p} ({session}) -- pztz labels {src}")
          path = os.path.join(labels, "anatomical_delineations", "zonal_pz_tz", "AI", src, f"{p}_{session}.nii.gz")
          if os.path.isfile(path): # if not, means we have no data
            prostate = nib.load(path)
            X = prostate.get_fdata().astype(np.uint8)
            fn_base = os.path.join(outdir,f"{p}-{session_num+1:02d}-0001")
            for slice in range(min_slice,max_slice):
              if slice < X.shape[-1]: # pztz may have fewer slices
                # PZTZ mask
                fn = fn_base + f"-{slice:04d}-pztz:{src}"
                save_npy_and_png_if_changed(fn, X[:,:,slice], 8)

def get_prostate_slices(labels, p, session, all_slices):
  """Determine slice range of prostate for a patient/session.

  Args:
      labels (str): Path to labels folder
      p (str): Patient ID
      session (str): Session ID
      all_slices (bool): If True, return full slice range

  Returns:
      tuple: (min_slice, max_slice) or (None, None) if no prostate data
  """
  # Determine slice range of prostate for patient p/session; if all_slices report full slice range.
  # Consider both AI whole-gland sources; merge slice spans when both exist.
  span = None  # (lo, hi) inclusive indices, or None if no mask slice found yet
  for src in ['Bosma22b', 'Guerbet23']:
    path = os.path.join(labels, "anatomical_delineations", "whole_gland", "AI", src, f"{p}_{session}.nii.gz")
    if not os.path.isfile(path):
      continue
    prostate = nib.load(path)
    X = prostate.get_fdata().astype(np.uint8)
    if all_slices:
      return 0, X.shape[-1]
    v_min = X.shape[-1]
    v_max = 0
    for s in range(0, X.shape[-1]):
      cnt = np.unique(X[:,:,s]).shape[0]
      if cnt > 1:
        if s < v_min:
          v_min = s
        if s > v_max:
          v_max = s
    if v_max >= v_min:
      if span is None:
        span = (v_min, v_max)
      else:
        lo, hi = span
        span = (min(lo, v_min), max(hi, v_max))
  if span is None:
    return None, None
  lo, hi = span
  return lo, hi + 1

if __name__ == '__main__':
  main()
