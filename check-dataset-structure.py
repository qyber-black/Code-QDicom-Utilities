#!/usr/bin/env python3
#
# check-dataset-structure.py - QDicom Utilities
#
# SPDX-FileCopyrightText: Copyright (C) 2021 Frank C Langbein <frank@langbein.org>, Cardiff University
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Check and possibly fix file system structure of a dicom dataset.
#
# This is quite specific to the Qyber\black archive structure, but
# may help to set this up for other archives as well and sort the
# data. This checks the file system hierarchy to follo our
# GROUP/PATIENT/SESSION/SCAN/DICOM_FILE hierarchy and also checks for
# duplicates and potential inconsistentcies. GROUP is arbitary, but often
# the year. PATIENT is a patient idetnified, expected to start with P and
# followed by a zero-filled number. SESSION is a zero-filled 2-digit number
# SCAN a zero-filled 4-digit number and DICOM_FILE has the filename
# PATIENT-SESSION-SCAN.IMA. It's mostly for Siemens dicom files.
#
# Usage example:
#
# ./check-dataset-structure.py -v scans
#
# See --help for full arguments

import os
import sys
import subprocess
import argparse

import numpy as np
import xxhash

sys.path.insert(1, os.path.dirname(os.path.realpath(__file__)))
from read_dicom_siemens import read_dicom

class colors:
  OK   = '\033[92m'
  WARN = '\033[93m'
  FAIL = '\033[91m'
  END  = '\033[0m'

bitmap_hash = {}
xxh = xxhash.xxh64()

def main():
  # Parse arguments and initial folder scan

  # Arguments
  parser = argparse.ArgumentParser()
  parser.add_argument('path', type=str,
                      help='base directory of dataset (of GROUP/PATIENT/SESSION/SCAN hierarchy).')
  parser.add_argument('-f', '--fix', action='store_true',
                      help='fix inconsistencies, if possible, without deleting anything')
  parser.add_argument('-v', '--verbose', action='count',
                      help='increase output verbosity', default=0)
  args = parser.parse_args()

  if not os.path.isdir(args.path):
    print("%s: not a directory = '%s'" % (parser.prog,args.path), file=sys.stderr)
    exit(1)

  # Start folder scan
  dir = os.path.normpath(args.path)
  if args.verbose > 0:
    print("Checking data archive: " + os.path.basename(dir))
  exit(check_scans(dir, "  ", args.fix, args.verbose))

def check_scans(dir, indent, fix, verbose):
  # Check MRI image scan hierarchy
  err = 0
  patients = []
  for fn in sorted(os.listdir(dir)):
    path = os.path.join(dir,fn)
    if os.path.isdir(path):
      new_patients, ne = check_group(path, indent, fix, verbose)
      err += ne
      for np in new_patients:
        if np in patients:
          err += 1
          print(("%s* %s: " + colors.FAIL + "fail: duplicate patient id:"
                 + colors.END + np) % (indent,fn))
        else:
          patients.append(np)
    else:
      err += 1
      print(("%s* %s: " + colors.FAIL + "fail: file in scans folder" + colors.END)
            % (indent,fn))
  if verbose > 0:
    print(("\n%s* total patients: " + colors.OK + "%d" + colors.END)
          % (indent,len(patients)))
  return err

def check_group(dir, indent, fix, verbose):
  # Check patients in MRI scan group hierarchy
  err = 0
  patients = []
  for fn in sorted(os.listdir(dir)):
    path = os.path.join(dir,fn)
    if os.path.isdir(path):
      if fn[0] != "P":
        err += 1
        print(("%s* %s/%s: " + colors.FAIL + "fail: not a patient id" + colors.END)
              % (indent,os.path.basename(dir),fn))
      else:
        if "-" in fn:
          fn_fix = fn.split("-")[0]
          if fn_fix != fn or True:
            err += 1
            print(("%s %s/%s: " + colors.FAIL + "fail: fix name" + colors.END
                  + " -> %s") % (indent,os.path.basename(dir),fn,fn_fix))
            if fix and len(fn_fix) > 0:
              subprocess.run(["git", "mv", path, os.path.join(dir,fn_fix)])
              fn = fn_fix
              path = os.path.join(dir,fn_fix)
        err += check_patient(path, fn, indent, fix, verbose)
        patients.append(fn)
    else:
      err += 1
      print(("%s* %s/%s: " + colors.FAIL + "fail: file not allowed here" + colors.END)
            % (indent,os.path.basename(dir),fn))
  return patients, err

def check_patient(dir, patient, indent, fix, verbose):
  # Check sessions in patient hierarchy
  err = 0
  files = []
  dirs = []
  for fn in sorted(os.listdir(dir)):
    path = os.path.join(dir,fn)
    if os.path.isdir(path):
      dirs.append(fn)
    elif os.path.isfile(path):
      files.append(fn)
    else:
      raise Exception("Unknown file type: %s" % path)
  if len(files) > 0:
    if len(dirs) > 0:
      err += 1
      print(("%s* %s" + colors.FAIL + "fail: mixed files and folders" + colors.END)
            % (indent,patient))
      check_bitmaps(dir, indent, verbose)
    else:
      err += 1
      n = 1
      new_session_name = ("%02" % n)
      session_path = os.path.join(dir,nmew_session_name)
      while os.path.isdir(session_path):
        n += 1
        new_session_name = ("%02" % n)
        session_path = os.path.join(dir,nmew_session_name)
      print(("%s* %s" + colors.FAIL + "fail: no session folder" + colors.END + " -> mv %s/XXXX")
            % (indent,patient,new_session_name))
      if fix:
        if not os.path.isdir(session_path):
          subprocess.run(["mkdir", session_path])
          subprocess.run(["git", "add", session_path])
        scan_folders = []
        for f in files:
          if len(fs) > 4:
            fs = f.split(".")
            if fs[2][0:7] == "PELVIS_":
              sf = fs[3]
              f_fix = patient+"-"+new_session_name+"-"+sf+"-"+fs[4]+"."+fs[-1]
            else:
              sf = fs[2]
              f_fix = patient+"-"+new_session_name+"-"+sf+"-"+fs[3]+"."+fs[-1]
            if sf not in scan_folders:
              if not os.path.isdir(os.path.join(session_path,sf)):
                subprocess.run(["mkdir", os.path.join(session_path,sf)])
                subprocess.run(["git", "add", os.path.join(session_path,sf)])
              scan_folders.append(sf)
            subprocess.run(["git", "mv", os.path.join(dir,f),
                           os.path.join(session_path,sf,f_fix)])
          else:
            print(("%s* %s" + colors.FAIL + "fail: strange name: %s" + colors.END)
                  % (indent,patient,f))
        err += check_session(session_path, patient, new_session_name, indent, fix, verbose)
  elif len(dirs) > 0:
    for d in dirs:
      if "-" in d:
        err += 1
        print(("%s* %s/%s: " + colors.FAIL + "fail: multiple session instances" + colors.END)
              % (indent,patient,d))
        check_bitmaps(dir, indent, verbose)
      else:
        err += check_session(os.path.join(dir,d), patient, d, indent, fix, verbose)
  else:
    err += 1
    print(("%s* %s: " + colors.FAIL + "fail: empty patient folder" + colors.END) % (indent,dir))
  return err

def check_session(dir, patient, session, indent, fix, verbose):
  # Check scans in session hierarchy
  err = 0
  ref_rec = {}
  try:
    if len(session) != 2:
      raise Exception("length")
    v = int(session)
  except:
    print(("%s* %s/%s: " + colors.WARN + "warn: session-id should be two-digit, zero-filled"
          + colors.END) % (indent,patient,session))
  files = []
  dirs = []
  for fn in sorted(os.listdir(dir)):
    path = os.path.join(dir,fn)
    if os.path.isdir(path):
      dirs.append(fn)
    elif os.path.isfile(path):
      files.append(fn)
    else:
      raise Exception("Unknown file type: %s" % path)
  if len(files) > 0:
    if len(dirs) > 0:
      err += 1
      print(("%s* %s/%s: " + colors.FAIL + "fail: mixed files and folders"+ colors.END)
            % indent)
      check_bitmaps(dir, indent, verbose)
    else:
      err += 1
      print(("%s* %s/%s: " + colors.FAIL + "fail: files in session folder" + colors.END
            + "-> mv XXXX") % (indent,patient,session))
      if fix:
        scan_folders = []
        for f in files:
          fs = f.split(".")
          if len(fs) > 4:
            if fs[2][0:7] == "PELVIS_":
              sf = fs[3]
              f_fix = patient+"-"+session+"-"+sf+"-"+fs[4]+"."+fs[-1]
            else:
              sf = fs[2]
              f_fix = patient+"-"+session+"-"+sf+"-"+fs[3]+"."+fs[-1]
            if sf not in scan_folders:
              if not os.path.isdir(os.path.join(dir,sf)):
                subprocess.run(["mkdir", os.path.join(dir,sf)])
                subprocess.run(["git", "add", os.path.join(dir,sf)])
              scan_folders.append(sf)
            subprocess.run(["git", "mv", os.path.join(dir,f), os.path.join(dir,sf,f_fix)])
          else:
            print(("%s* %s/%s: " + colors.FAIL + "fail: strange name: %s" + colors.END)
                  % (indent,f,patient,session))
        for d in scan_folders:
          ne, rec = check_scan(os.path.join(dir,d), patient, session,
                               d, indent, fix, verbose)
          if ne == 0:
            if ref_rec == {}:
              ref_rec = rec
            else:
              for key in rec:
                if key in ref_rec:
                  if (d[0] != '5' or
                      (d[0] == '5' and key != 'ContentDate' and
                                       key != 'RequestingPhysician' and
                                       key != 'InstanceCreationDate' and
                                       key != 'SeriesDate')):
                    if rec[key] != ref_rec[key]:
                      if (key == 'StudyTime' or
                          key == 'AcquisitionDate' or
                          key == 'StudyDate' or
                          key == 'StudyTime' or
                          key == 'ContentDate' or
                          key == 'InstanceCreationDate' or
                          key == 'SeriesDate' or
                          key == 'RequestingPhysician'):
                        mode = colors.WARN
                      else:
                        mode = colors.FAIL
                        err += 1
                      if mode == colors.FAIL or verbose > 0:
                        print(("%s* %s/%s/%s: " + mode + "fail: diff across scan " + str(key)
                               + ": " + str(rec[key]) + " != " + str(ref_rec[key]) + colors.END)
                               % (indent,patient,session,d))
                else:
                  ref_rec[key] = rec[key]
          err += ne
  elif len(dirs) > 0:
    for d in dirs:
      if "_" in d:
        ds = d.split("_")
        if len(ds) > 1:
          err += 1
          print(("%s* %s/%s/%s: " + colors.FAIL + "fail: long scan folder name" + colors.END
                 + " -> %s") % (indent,patient,session,d,ds[-1]))
          if fix:
            subprocess.run(["git", "mv", os.path.join(dir,d), os.path.join(dir,ds[-1])])
            d = ds[-1]
      ne, rec = check_scan(os.path.join(dir,d), patient, session, d, indent, fix, verbose)
      if ne == 0:
        if ref_rec == {}:
          ref_rec = rec
        else:
          for key in rec:
            if key in ref_rec:
              if (d[0] != '5' or
                  (d[0] == '5' and key != 'ContentDate' and
                                   key != 'InstanceCreationDate' and
                                   key != 'RequestingPhysician' and
                                   key != 'SeriesDate')):
                if rec[key] != ref_rec[key]:
                  if (key == 'StudyTime' or
                      key == 'AcquisitionDate' or
                      key == 'StudyDate' or
                      key == 'StudyTime' or
                      key == 'ContentDate' or
                      key == 'InstanceCreationDate' or
                      key == 'SeriesDate' or
                      key == 'RequestingPHysician'):
                    mode = colors.WARN
                  else:
                    mode = colors.FAIL
                    err += 1
                  if mode == colors.FAIL or verbose > 0:
                    print(("%s* %s/%s/%s: " + mode + "fail: diff across scan " + str(key)
                           + ": " + str(rec[key]) + " != " + str(ref_rec[key]) + colors.END)
                           % (indent,patient,session,d))
            else:
              ref_rec[key] = rec[key]
      err += ne
  else:
    err += 1
    print(("%s" + colors.FAIL + "fail: empty session folder" + colors.END) % indent)
  if verbose > 0 and "PatientName" in ref_rec:
    l = min(len(ref_rec['PatientName']),len(patient))
    if ref_rec['PatientName'][0:l] != patient[0:l]:
      print(("%s* %s/%s: " + colors.WARN + "warn: patient name record '%s' does not match filename '%s'"
            + colors.END) % (indent,patient,session,ref_rec['PatientName'],patient))
  return err

def check_scan(dir, patient, session, scan, indent, fix, verbose):
  # Check dicoms in scan
  err = 0
  try:
    if len(scan) != 4:
      raise Exception("length")
    v = int(scan)
  except:
    print(("%s* %s/%s/%s: " + colors.WARN + "warn: scan-id should be four-digit, zero-filled"
          + colors.END) % (indent,patient,session,scan))
  files = []
  dirs = []
  for fn in sorted(os.listdir(dir)):
    path = os.path.join(dir,fn)
    if os.path.isdir(path):
      dirs.append(fn)
    elif os.path.isfile(path):
      files.append(fn)
    else:
      raise Exception("Unknown file type: %s" % fnp)
  if len(dirs) > 0:
    err += 1
    print(("%s* %s/%s/%s: " + colors.FAIL + "fail: folders in scan folder" + colors.END)
          % (indent,patient,session,scan))
  elif len(files) > 0:
    base = patient+"-"+session+"-"+scan
    for f in files:
      if "-" in f:
        if f[0:len(base)] == base:
          fs = f.split("-")
          ok = False
          if len(fs) == 4:
            fss = fs[3].split(".")
            if len(fss) == 2:
              if fss[1] == "IMA":
                try:
                  if len(fss[0]) != 4:
                    raise Exception("length")
                  v = int(fss[0])
                  ok = True
                except:
                  pass
              elif fss[1] == "json":
                if fss[0][-17:] == "_annotations.json":
                  ok = True
              elif fss[1] == "SR":
                ok = True
          elif len(fs) == 3:
            if fs[2][-23:] == "_scan_annotations.json":
              ok = True
          if not ok:
            err += 1
            print(("%s* %s/%s/%s%s: " + colors.FAIL + "fail: broken filename" + colors.END)
                  % (indent,patient,session,scan,f))
        else:
          err += 1
          print(("%s* %s/%s/%s/%s: " + colors.FAIL + "fail: inconsistent filename" + colors.END)
                % (indent,patient,session,scan,f))
          fs = f.split("-")
          if len(fs) == 4:
            f_fix = patient+"-"+session+"-"+scan+"-"+fs[3]
            print("%s  -> %s" % (indent,f_fix))
            if fix:
              subprocess.run(["git", "mv", os.path.join(dir,f), os.path.join(dir,f_fix)])
      elif "." in f:
        err += 1
        print(("%s* %s/%s/%s/%s: " + colors.FAIL + "fail: broken filename" + colors.END)
              % (indent,patient,session,scan,f))
        fs = f.split(".")
        if len(fs) > 4:
          if fs[2][0:7] == "PELVIS_":
            f_fix = patient+"-"+session+"-"+scan+"-"+fs[4]+"."+fs[-1]
          else:
            f_fix = patient+"-"+session+"-"+scan+"-"+fs[3]+"."+fs[-1]
          print("%s  -> %s" % (indent,f_fix))
          if fix:
            subprocess.run(["git", "mv", os.path.join(dir,f), os.path.join(dir,f_fix)])
      else:
        err += 1
        print(("%s* %s/%s/%s/%s: " + colors.FAIL + "fail: unknown filename" + colors.END)
              % (indent,patient,session,scan,f))
  else:
    err += 1
    print(("%s* %s/%s/%s: " + colors.FAIL + "fail: empty scan folder" + colors.END)
          % (indent,patient,session,scan))
  ne, ref_rec = check_dicoms(dir, patient, session, scan, indent, fix, verbose)
  err += ne
  return err, ref_rec

def check_dicoms(dir, patient, session, scan, indent, fix, verbose):
  err = 0
  dirs = []
  files = []
  for fn in sorted(os.listdir(dir)):
    path = os.path.join(dir,fn)
    if os.path.isdir(path):
      dirs.append(fn)
    elif os.path.isfile(path):
      if fn[-3:] == ".SR":
        pass
      else:
        files.append(fn)
    else:
      raise Exception("Unknown file type: %s" % fnp)

  ref_rec = {}
  series_time_col = None
  out_base = indent+":"+patient+"/"+session+"/"+scan+"/"
  out_len = 0
  for f in sorted(files):
    out = out_base+f
    if verbose > 0:
      print(out+" "*(max(0,out_len-len(out))),end='\r',flush=True)
    out_len = len(out)
    path = os.path.join(dir, f)
    dicom, info = read_dicom(path)
    xxh.reset()
    try:
      arr = dicom.pixel_array.astype(np.uint16, order='C', casting='safe')
    except Exception as e:
      arr = None
    if arr is None:
      try:
        arr = dicom[(0x7fe1, 0x1010)].value
      except Exception as e:
        print(("%s* %s/%s/%s/%s: " + colors.WARN + "warn: no pixel/spectrum data" + colors.END)
              % (indent,patient,session,scan,f))
        arr = None
    if arr is not None:
      overlay = {}
      for idx in range(0,0x1f):
        try:
          overlay[idx] = dicom.overlay_array(0x6000+idx)
        except:
          pass
      xxh.update(arr)
      h = str(xxh.hexdigest())
      if h in bitmap_hash:
        if path not in bitmap_hash[h]:
          for bm in bitmap_hash[h]:
            d2, i2 = read_dicom(bm)
            o2 = {}
            for idx in range(0,0x1f):
              try:
                o2[idx] = d2.overlay_array(0x6000+idx)
              except:
                pass
            try:
              arr2 = d2.pixel_array.astype(np.uint16, order='C', casting='safe')
            except:
              arr2 = None
            if arr2 is None:
              try:
                arr2 = dicom[(0x7fe1, 0x1010)].value
              except:
                arr2 = None
            if np.array_equal(arr,arr2):
              equal = True
              ol = ""
              if len(o2) == len(overlay):
                for k in o2:
                  if k in overlay:
                    if np.array_equal(o2[k],overlay[k]):
                      ol += " "+str(k)
                    else:
                      equal = False
                      break
                  else:
                    equal = False
                    break
                if ol != "":
                  ol = "[overlays: " + ol + "]"
              else:
                equal = False
              if equal:
                print(("%s* %s: " + colors.FAIL + "fail: == %s %s"
                      + colors.END) % (indent,f, bm, ol))
          bitmap_hash[h].append(path)
      else:
        bitmap_hash[h] = [path]

    rec = {}
    instance = None
    for rf in ['AcquisitionDate',
               'ContentDate',
               'InstanceCreationDate',
               'SeriesDate',
               'SeriesTime',
               'StudyDate',
               'StudyTime',
               'ReferringPhysicianName',
               'PatientName',
               'PatientID',
               'IssuerOfPatientID',
               'MilitaryRank',
               'BranchOfService',
               'PatientReligiousPreference',
               "[Patient's Name]",
               'RequestingPhysician',
               'RequestingService',
               'RequestedProcedureDescription',
               "[Referring Physician's Name]",
               "Private tag data",
               "AdmissionID",
               "CurrentPatientLocation"]:
      if rf in info:
        val = info[rf]
        if (rf == "RequestedProcedureDescription" and
            val == "MRI Pelvis prostate"):
           val = "pelvis_0001 Uni of Swansea"
        rec[rf] = val
    if len(ref_rec) == 0:
      ref_rec = rec
    else:
      for key in rec:
        if key in ref_rec:
          if ref_rec[key] != rec[key]:
            if (scan[0] != '5' or
                (scan[0] == '5' and key != 'ContentDate' and
                                    key != 'InstanceCreationDate' and
                                    key != 'SeriesTime' and
                                    key != 'SeriesDate')):
              err += 1
              print(("%s* %s/%s/%s/%s: " + colors.FAIL
                     + "fail: scan records don't match across slices"
                     + colors.END) % (indent,patient,session,scan,f))
              print(("%s  " + str(key) + ": " + str(rec[key]) + " != "
                     + str(ref_rec[key])) % indent)
        else:
          if verbose > 0:
            print(("%s* %s: " + colors.WARN + "warn: extra scan record in slice"
                  + colors.END + " - " + key) % (indent,f))
          ref_rec[key] = rec[key]
    if 'InstanceNumber' in info:
      if instance == None:
        instance = info['InstanceNumber']
      else:
        if instance < info['InstanceNumber']:
          instance = info['InstanceNumber']
        else:
          err += 1
          print(("%s* %s/%s/%s/%s: " + colors.FAIL + "fail: instance number not increasing"
                + colors.END) % (indent,patient,session,scan,f))
  if "SeriesTime" in ref_rec:
    del ref_rec["SeriesTime"]
  return err, ref_rec

def check_bitmaps(dir, indent, verbose):
  # Collect bitmaps from files where errors occured, to find duplicates
  dirs = []
  files = []
  for fn in sorted(os.listdir(dir)):
    path = os.path.join(dir,fn)
    if os.path.isdir(path):
      dirs.append(fn)
    elif os.path.isfile(path):
      files.append(fn)
    else:
      raise Exception("Unknown file type: %s" % fnp)

  out_base = indent+"#"+dir+"/"
  out_len = 0
  for f in sorted(files):
    out = out_base+f
    if verbose > 0:
      print(out+" "*(max(0,out_len-len(out))),end='\r',flush=True)
    out_len = len(out)
    path = os.path.join(dir, f)
    dicom, info = read_dicom(path)
    xxh.reset()
    try:
      arr = dicom.pixel_array.astype(np.uint16, order='C', casting='safe')
      xxh.update(arr)
      h = str(xxh.hexdigest())
      if h in bitmap_hash and path not in bitmap_hash[h]:
        bitmap_hash[h].append(path)
      else:
        bitmap_hash[h] = [path]
    except:
      pass

  for d in dirs:
    check_bitmaps(os.path.join(dir,d), indent, verbose)

if __name__ == '__main__':
  main()
