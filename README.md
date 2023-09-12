# text-generation-webui-edge-tts
A simple extension for the [text-generation-webui by oobabooga](https://github.com/oobabooga/text-generation-webui) that uses [edge_tts](https://github.com/rany2/edge-tts) for audio output.

## How to install
Assuming you already have the webui set up:

1. Activate the conda environment with the `cmd_xxx.bat` or using `conda activate textgen`
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

## Note
Edge TTS is a free API provided by Microsoft. An internet connection is required for the TTS to function.