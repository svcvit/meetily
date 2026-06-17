!macro NSIS_HOOK_POSTINSTALL
  CopyFiles "$INSTDIR\resources\sherpa-dlls\*.dll" "$INSTDIR"
!macroend

!macro NSIS_HOOK_POSTUNINSTALL
  Delete "$INSTDIR\onnxruntime.dll"
  Delete "$INSTDIR\sherpa-onnx-c-api.dll"
  Delete "$INSTDIR\kaldi-native-fbank-core.dll"
  Delete "$INSTDIR\kaldi-decoder-core.dll"
  Delete "$INSTDIR\sherpa-onnx-kaldifst-core.dll"
  Delete "$INSTDIR\sherpa-onnx-fstfar.dll"
  Delete "$INSTDIR\sherpa-onnx-fst.dll"
  Delete "$INSTDIR\kissfft-float.dll"
  Delete "$INSTDIR\piper_phonemize.dll"
  Delete "$INSTDIR\espeak-ng.dll"
  Delete "$INSTDIR\ucd.dll"
  Delete "$INSTDIR\ssentencepiece_core.dll"
!macroend
