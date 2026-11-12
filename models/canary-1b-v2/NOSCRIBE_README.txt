The "canary-1b-v2" model is used by the "Local Canary" backend plugin
(noScribe/plugins/local_canary). It runs NVIDIA Canary 1B v2 on the CPU with
ONNX Runtime via the onnx-asr package. No GPU is needed. Canary supports 25
European languages (bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt,
lv, mt, nl, pl, pt, ro, ru, sk, sl, sv, uk). It cannot detect the language:
choose the spoken language in noScribe. With "Auto" the plugin uses
fallback_language from plugin-settings/local-canary.yml (default: nl) and
says so in the log.

The model appears in the model menu only when all files below are present.
The weights are not part of the git repository.

Files needed in this folder (int8 quantization, about 1.03 GB in total)
------------------------------------------------------------------------
From https://huggingface.co/istupakov/canary-1b-v2-onnx/tree/main
(tested revision 5ebc1520cef7b6b318b3526ad17adbfe00bc1bfc):

    config.json                  68 bytes   (marker: model_type nemo-conformer-aed)
    vocab.txt                    208 KB
    encoder-model.int8.onnx      859 MB
    decoder-model.int8.onnx      170 MB

From https://huggingface.co/istupakov/silero-vad-onnx/tree/main
(tested revision b3e3ee3cce4c11ceb63b1a0b229d916069c1ddf6), into the
subfolder "silero-vad":

    silero-vad/silero_vad.onnx   2.3 MB

Instead of the int8 files you may use the float32 export
(encoder-model.onnx + encoder-model.onnx.data + decoder-model.onnx, about
3.9 GB). It is slower on CPU; int8 is used whenever both are present.

Download with Python (pip install huggingface_hub):

    from huggingface_hub import hf_hub_download
    d = "models/canary-1b-v2"
    for f in ["config.json", "vocab.txt", "encoder-model.int8.onnx", "decoder-model.int8.onnx"]:
        hf_hub_download("istupakov/canary-1b-v2-onnx", f, local_dir=d)
    hf_hub_download("istupakov/silero-vad-onnx", "silero_vad.onnx", local_dir=d + "/silero-vad")

Afterwards you may delete the ".cache" folders that huggingface_hub creates.
A model folder with the same layout can also be placed in the user models
directory under another name.

Model licenses
--------------
Canary 1B v2
  Original model: https://huggingface.co/nvidia/canary-1b-v2
  Copyright NVIDIA Corporation. Licensed under the Creative Commons
  Attribution 4.0 International License (CC BY 4.0):
  https://creativecommons.org/licenses/by/4.0/
  The files in this folder are an ONNX conversion of that model by Ilya
  Stupakov (https://huggingface.co/istupakov/canary-1b-v2-onnx), published
  under the same CC BY 4.0 license. The conversion changes the file format
  and quantizes the weights to int8; it does not retrain the model.

Silero VAD
  Original model: https://github.com/snakers4/silero-vad
  Copyright (c) 2020-present Silero Team. Licensed under the MIT License.
  ONNX file obtained from https://huggingface.co/istupakov/silero-vad-onnx.
