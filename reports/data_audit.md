# Dataset Audit Report

Status: **review**

## Summary
- Total rows: 7441
- Total speakers: 91
- Split counts: train=5228, val=1066, test=1147
- Emotion counts: {"angry": 1271, "disgust": 1271, "fear": 1271, "happy": 1271, "neutral": 1087, "sad": 1270}
- Speaker clip count range: 76 to 82
- Duration seconds: min=1.267937, p50=2.5025, p95=3.470125, max=5.005

## Audit Checks
- Missing files: 0
- Unreadable files: 0
- Manifest/header mismatches: 0
- Speaker overlap: {"train_test": 0, "train_val": 0, "val_test": 0}
- Duplicate manifest paths: 0
- Duplicate clip IDs: 0
- Exact duplicate audio groups: 3
- Duration outliers: 163

## Findings
- Exact duplicate audio groups found: 3

## Duration Outlier Samples
- 1002_ITH_DIS_XX (train): 3.803812 sec
- 1003_ITH_DIS_XX (train): 4.037375 sec
- 1003_ITH_NEU_XX (train): 4.037375 sec
- 1003_IWL_DIS_XX (train): 4.404375 sec
- 1003_IWL_HAP_XX (train): 3.903875 sec
- 1003_IWL_NEU_XX (train): 3.93725 sec
- 1003_MTI_DIS_XX (train): 4.237563 sec
- 1003_TIE_DIS_XX (train): 4.270937 sec
- 1003_TIE_SAD_XX (train): 4.371062 sec
- 1003_TSI_ANG_XX (train): 3.970625 sec
- 1004_IEO_ANG_HI (val): 4.004 sec
- 1004_IEO_DIS_HI (val): 4.838188 sec
- 1004_IEO_SAD_HI (val): 5.005 sec
- 1004_IEO_SAD_MD (val): 4.037375 sec
- 1004_IWL_NEU_XX (val): 3.903875 sec
- 1004_TAI_DIS_XX (val): 3.903875 sec
- 1004_TIE_DIS_XX (val): 4.337688 sec
- 1005_MTI_DIS_XX (train): 3.93725 sec
- 1005_TIE_HAP_XX (train): 4.204187 sec
- 1005_TIE_SAD_XX (train): 4.704688 sec

## Duplicate Audio Samples
- ['/mnt/c/Desktop/Projects/speech emotion detection system/AudioWAV/1006_TIE_HAP_XX.wav', '/mnt/c/Desktop/Projects/speech emotion detection system/AudioWAV/1006_TIE_NEU_XX.wav']
- ['/mnt/c/Desktop/Projects/speech emotion detection system/AudioWAV/1013_WSI_DIS_XX.wav', '/mnt/c/Desktop/Projects/speech emotion detection system/AudioWAV/1013_WSI_SAD_XX.wav']
- ['/mnt/c/Desktop/Projects/speech emotion detection system/AudioWAV/1017_IWW_ANG_XX.wav', '/mnt/c/Desktop/Projects/speech emotion detection system/AudioWAV/1017_IWW_FEA_XX.wav']
