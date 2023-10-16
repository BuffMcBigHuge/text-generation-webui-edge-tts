# edge_tts for text-generation-webui

import time
from pathlib import Path
import random
import sys
import os
import traceback
import librosa
import html

import gradio as gr
import torch
import edge_tts
import asyncio
from scipy.io import wavfile
import numpy as np
from threading import Thread

from modules import chat, shared, ui_chat
from modules.utils import gradio
from modules.logging_colors import logger

from fairseq import checkpoint_utils

sys.path.append("extensions/edge_tts")

import tts_preprocessor
from rmvpe import RMVPE
from vc_infer_pipeline import VC
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from config import Config

torch._C._jit_set_profiling_mode(False)

params = {
    'activate': True,
    'speaker': None,
    'language': 'en',
    'show_text': False,
    'autoplay': False,
    'rvc': False,
    'rvc_model': None,
    'transpose': 2,
    'index_rate': 1,
    'protect': 0.33
}
current_params = params.copy()

voices = []
rvc_models = []
rvc_config = Config()
hubert_model = None
rmvpe_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["extensions/edge_tts/models/hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(rvc_config.device)
    if rvc_config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


def get_all_paths(relative_directory, filetype=None):
    folders = []
    files = []

    for dirpath, dirnames, filenames in os.walk(relative_directory):
        for dirname in dirnames:
            folders.append(os.path.relpath(os.path.join(dirpath, dirname), relative_directory))
        for filename in filenames:
            if filetype is None or filename.endswith(filetype):
                files.append(os.path.relpath(os.path.join(dirpath, filename), relative_directory))

    return folders, files


def remove_tts_from_history(history):
    for i, entry in enumerate(history['internal']):
        history['visible'][i] = [history['visible'][i][0], entry[1]]

    return history


def toggle_text_in_history(history):
    for i, entry in enumerate(history['visible']):
        visible_reply = entry[1]
        if visible_reply.startswith('<audio'):
            if params['show_text']:
                reply = history['internal'][i][1]
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>\n\n{reply}"]
            else:
                history['visible'][i] = [history['visible'][i][0], f"{visible_reply.split('</audio>')[0]}</audio>"]

    return history


def state_modifier(state):
    if not params['activate']:
        return state

    state['stream'] = False
    return state


def input_modifier(string, state):
    if not params['activate']:
        return string

    shared.processing_message = "*Is recording a voice message...*"
    return string


def history_modifier(history):
    # Remove autoplay from the last reply
    if len(history['internal']) > 0:
        history['visible'][-1] = [
            history['visible'][-1][0],
            history['visible'][-1][1].replace('controls autoplay>', 'controls>')
        ]

    return history


def output_modifier(string, state):
    if not params['activate']:
        return string
    
    if params['speaker'] is None:
        return logger.error('No speaker selected')
    
    if params['rvc'] is True and params['rvc_model'] is None:
        return logger.error('No RVC model selected')

    original_string = string

    original_string = string
    string = tts_preprocessor.replace_invalid_chars(html.unescape(string))
    string = tts_preprocessor.replace_abbreviations(string)
    string = tts_preprocessor.clean_whitespace(string)

    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = Path(f'extensions/edge_tts/outputs/{int(time.time())}.mp3')

        print(f'Outputting audio to {str(output_file)}')
        # print(f'{string}')

        communicate = edge_tts.Communicate(string, params['speaker'])
        asyncio.run(communicate.save(output_file))

        if (params['rvc'] is True):
            print('Running RVC')
            audio = tts(output_file, params['rvc_model'], params['transpose'], 'rmvpe', params['index_rate'], params['protect'])
            wavfile.write(output_file, 44100, audio.astype(np.int16))
        
        autoplay = 'autoplay' if params['autoplay'] else ''
        string = f'<audio src="file/{output_file.as_posix()}" controls {autoplay}></audio>'
        if params['show_text']:
            string += f'\n\n{original_string}'

    shared.processing_message = "*Is typing...*"
    return string


def random_sentence():
    with open(Path("extensions/edge_tts/harvard_sentences.txt")) as f:
        return random.choice(list(f))


def voice_preview(preview_text):    
    global params

    if params['speaker'] is None:
        return logger.error('No speaker selected')
    
    if params['rvc'] is True and params['rvc_model'] is None:
        return logger.error('No RVC model selected')

    string = preview_text or random_sentence()

    output_file = Path('extensions/edge_tts/outputs/voice_preview.mp3')

    communicate = edge_tts.Communicate(string, params['speaker'])
    asyncio.run(communicate.save(output_file))

    if (params['rvc'] is True):
        audio = tts(output_file, params['rvc_model'], params['transpose'], 'rmvpe', params['index_rate'], params['protect'])
        wavfile.write(output_file, 44100, audio.astype(np.int16))

    return f'<audio src="file/{output_file.as_posix()}?{int(time.time())}" controls autoplay></audio>'


def refresh(x): 
    global voices, current_params

    for i in params:
        if params[i] != current_params[i]:
            current_params = params.copy()
            break

    # Get Voices
    voices = asyncio.run(edge_tts.list_voices())
    print(f"Loaded {len(voices)} voices.")
    voices = [x['ShortName'] for x in voices]
    
    # Get RVC Models
    folders, files = get_all_paths('extensions/edge_tts/rvc_models', '.pth')
    rvc_models = files
    print(f"Found {len(rvc_models)} rvc models.")

    if params['speaker'] not in voices:
        params['speaker'] = 'en-US-MichelleNeural'

    return [gr.update(value=params['speaker'], choices=voices), gr.update(value=params['rvc_model'], choices=rvc_models)]


def setup():
    global voices, current_params, rvc_models, rmvpe_model, hubert_model

    print("Loading hubert model...")
    hubert_model = load_hubert()
    print("Hubert model loaded.")

    print("Loading rmvpe model...")
    rmvpe_model = RMVPE("extensions/edge_tts/models/rmvpe.pt", rvc_config.is_half, rvc_config.device)
    print("rmvpe model loaded.")

    # Cannot run async on main gradio thread
    # This works, but does not refresh gradio
    thread = Thread(target=refresh, args=(None,))
    thread.start()


def ui():
    # Gradio elements
    with gr.Accordion("Edge TTS"):
        with gr.Row():
            activate = gr.Checkbox(value=params['activate'], label='Activate TTS')
            autoplay = gr.Checkbox(value=params['autoplay'], label='Play TTS automatically')
            refresh_button = gr.Button("Load Voices")

        show_text = gr.Checkbox(value=params['show_text'], label='Show message text under audio player')
        voice_dropdown = gr.Dropdown(value=params['speaker'], choices=voices, label='TTS voice')

        with gr.Row():
            rvc = gr.Checkbox(value=params['rvc'], label='Use RVC')
            model_dropdown = gr.Dropdown(value=params['rvc_model'], choices=rvc_models, label='RVC Model')

        with gr.Row():
            preview_text = gr.Text(show_label=False, placeholder="Preview text", elem_id="coqui_preview_text")
            preview_play = gr.Button("Preview")
            preview_audio = gr.HTML(visible=False)

        with gr.Column():
            transpose = gr.Slider(minimum=-12, maximum=12, value=params['transpose'], step=1, label='Transpose')
            index_rate = gr.Slider(minimum=0, maximum=1, value=params['index_rate'], step=0.01, label='Index Rate')
            protect = gr.Slider(minimum=0, maximum=0.5, value=params['protect'], step=0.01, label='Protect')

    if shared.is_chat():
        # Toggle message text in history
        show_text.change(
            lambda x: params.update({"show_text": x}), show_text, None).then(
            toggle_text_in_history, gradio('history'), gradio('history')).then(
            chat.save_history, gradio('history', 'unique_id', 'character_menu', 'mode'), None).then(
            chat.redraw_html, gradio(ui_chat.reload_arr), gradio('display'))
            
    # Event functions to update the parameters in the backend
    activate.change(lambda x: params.update({"activate": x}), activate, None)
    autoplay.change(lambda x: params.update({"autoplay": x}), autoplay, None)
    voice_dropdown.change(lambda x: params.update({"speaker": x}), voice_dropdown, None)
    rvc.change(lambda x: params.update({"rvc": x}), rvc, None)
    model_dropdown.change(lambda x: params.update({"rvc_model": x}), model_dropdown, None)
    transpose.change(lambda x: params.update({"transpose": x}), transpose, None)
    index_rate.change(lambda x: params.update({"index_rate": x}), index_rate, None)
    protect.change(lambda x: params.update({"protect": x}), protect, None)

    # Play preview
    preview_text.submit(voice_preview, preview_text, preview_audio)
    preview_play.click(voice_preview, preview_text, preview_audio)

    # Refresh voices
    refresh_button.click(refresh, refresh_button, [voice_dropdown, model_dropdown])

# RVC functions, retrieved via https://github.com/litagin02/rvc-tts-webui

def model_data(model_name):
    # global n_spk, tgt_sr, net_g, vc, cpt, version, index_file
    pth_path = f'extensions/edge_tts/rvc_models/{model_name}'
    
    cpt = torch.load(pth_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=True)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=True)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError("Unknown version")
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    print("Model loaded")
    net_g.eval().to('cuda')
    net_g = net_g.half()
    vc = VC(tgt_sr, rvc_config)
    # n_spk = cpt["config"][-3]

    index_file = ''

    return tgt_sr, net_g, vc, version, index_file, if_f0


def tts(
    output_file,
    model_name,
    f0_up_key=1,
    f0_method='rmvpe',
    index_rate=1,
    protect=0.33,
    filter_radius=3,
    resample_sr=0,
    rms_mix_rate=0.25,
):
    try:
        global hubert_model, rmvpe_model

        edge_output_filename = output_file

        tgt_sr, net_g, vc, version, index_file, if_f0 = model_data(model_name)
        
        audio, sr = librosa.load(edge_output_filename, sr=16000, mono=True)
        duration = len(audio) / sr

        print(f"Audio duration: {duration}s")
    
        f0_up_key = int(f0_up_key)

        if not hubert_model:
            load_hubert()
        if f0_method == "rmvpe":
            vc.model_rmvpe = rmvpe_model
        times = [0, 0, 0]
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            0,
            audio,
            edge_output_filename,
            times,
            f0_up_key,
            f0_method,
            index_file,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            None,
        )
        if tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        info = f"Success."
        print(info)
        return audio_opt
    except EOFError:
        info = (
            "It seems that the edge-tts output is not valid. "
            "This may occur when the input text and the speaker do not match. "
            "For example, maybe you entered Japanese (without alphabets) text but chose non-Japanese speaker?"
        )
        print(info)
    
    except:
        info = traceback.format_exc()
        print(info)
        return info, None, None