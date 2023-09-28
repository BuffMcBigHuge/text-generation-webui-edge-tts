# text-generation-webui-edge-tts
A simple extension for the [text-generation-webui by oobabooga](https://github.com/oobabooga/text-generation-webui) that uses [edge_tts](https://github.com/rany2/edge-tts) for audio output. It also supports post-processing using [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

## How to install
Assuming you already have the text-generation-webui set up (min version [1.6.1](https://github.com/oobabooga/text-generation-webui/releases/tag/1.6.1)):

1. Activate the conda environment using `conda activate textgen`
2. Enter the  `text-generation-webui/extensions/` directory and clone this repository
```
cd text-generation-webui/extensions/
git clone https://github.com/BuffMcBigHuge/text-generation-webui-edge-tts.git edge_tts/
```
3. Install the requirements
```
pip install -r edge_tts/requirements.txt
```
4. Add `--extensions edge_tts` to your startup script <br/> <b>or</b> <br/> enable it through the `Session` tab in the webui
5. Download the required RVC models and place them in the `extensions/edge_tts/models` folder
```
curl -L -O https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
curl -L -O https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt
```
6. Add your `.pth` files to `extensions/edge_tts/rvc_models`

## Notes
Edge TTS is a free API provided by Microsoft. An internet connection is required for the TTS to function.

RVC was inspired by [rvc-tts-webui](https://github.com/litagin02/rvc-tts-webui/tree/main)