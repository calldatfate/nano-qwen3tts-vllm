import gradio as gr
from nano_qwen3tts_vllm.interface import Qwen3TTSInterface
import soundfile as sf
import tempfile
import torch
import gc

# Global state for caching the model
current_model_name = None
interface = None
ENFORCE_EAGER = False

def load_model(model_name):
    global current_model_name, interface
    
    # If the requested model is already loaded, do nothing
    if interface is not None and current_model_name == model_name:
        return True, "Model already loaded."
        
    print(f"Loading model: {model_name}...")
    
    # Unload previous model to free up VRAM
    if interface is not None:
        interface.shutdown()
        interface = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    try:
        interface = Qwen3TTSInterface.from_pretrained(
            pretrained_model_name_or_path=model_name,
            enforce_eager=ENFORCE_EAGER,
            tensor_parallel_size=1,
        )
        current_model_name = model_name
        return True, f"Successfully loaded {model_name}"
    except Exception as e:
        interface = None
        current_model_name = None
        import traceback
        return False, f"Error loading model: {e}\n{traceback.format_exc()}"

def generate_tts(model_selection, text, instruction_text, speaker_selection, ref_audio_path, ref_text, language, temperature):
    global interface
    
    # 1. Load the model if it's different from the currently loaded one
    success, msg = load_model(model_selection)
    if not success:
        return None, msg

    try:
        # 2. Check which mode we are running based on the model name
        if "VoiceDesign" in model_selection:
            # VoiceDesign mode requires instruction
            audio_codes = list(interface.generate_voice_design(
                text=text,
                language=language,
                instruct=instruction_text,
                temperature=temperature,
            ))
        elif "CustomVoice" in model_selection:
            # CustomVoice mode uses predefined speakers (Vivian, Mike, etc.)
            audio_codes = list(interface.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker_selection,
                temperature=temperature,
            ))
        elif "Base" in model_selection:
            # Base mode requires reference audio
            if not ref_audio_path:
                return None, "Р”Р»СЏ РєР»РѕРЅРёСЂРѕРІР°РЅРёСЏ РіРѕР»РѕСЃР° РѕР±СЏР·Р°С‚РµР»СЊРЅРѕ Р·Р°РіСЂСѓР·РёС‚Рµ РѕСЂРёРіРёРЅР°Р»СЊРЅРѕРµ Р°СѓРґРёРѕ."
                
            if not ref_text.strip():
                try:
                    from faster_whisper import WhisperModel
                    # Load small model on GPU
                    whisper_model = WhisperModel("tiny", device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16" if torch.cuda.is_available() else "int8")
                    segments, info = whisper_model.transcribe(ref_audio_path, beam_size=5)
                    ref_text = " ".join([segment.text for segment in segments]).strip()
                    if not ref_text:
                        raise ValueError("Whisper СЂР°СЃРїРѕР·РЅР°Р» РїСѓСЃС‚СѓСЋ СЃС‚СЂРѕРєСѓ")
                except Exception as e:
                    return None, f"РћС€РёР±РєР° Р°РІС‚Рѕ-СЂР°СЃРїРѕР·РЅР°РІР°РЅРёСЏ С‚РµРєСЃС‚Р°: {e}. РџРѕР¶Р°Р»СѓР№СЃС‚Р°, РІРІРµРґРёС‚Рµ С‚РµРєСЃС‚ РІСЂСѓС‡РЅСѓСЋ."
            
            ref_audio, ref_sr = sf.read(ref_audio_path)
            prompt = interface.create_voice_clone_prompt(
                ref_audio=ref_audio,
                ref_sr=ref_sr,
                ref_text=ref_text,
                ref_language=language,
            )
            audio_codes = list(interface.generate_voice_clone(
                text=text,
                language=language,
                prompt=prompt,
                temperature=temperature,
            ))
        else:
            return None, "Model type not supported via this UI yet."
            
        # 3. Decode chunks to audio
        wavs, sr = interface.speech_tokenizer.decode([{"audio_codes": audio_codes}])
        
        # 4. Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)
        
        return temp_file.name, "РЈСЃРїРµС€РЅР°СЏ РіРµРЅРµСЂР°С†РёСЏ!"
    except Exception as e:
        import traceback
        return None, f"РћС€РёР±РєР° РїСЂРё РіРµРЅРµСЂР°С†РёРё: {str(e)}\n\n{traceback.format_exc()}"

def update_ui_for_model(model_name):
    """Dynamically change visibility depending on the chosen model"""
    show_instruction = "VoiceDesign" in model_name
    show_speaker = "CustomVoice" in model_name
    show_clone = "Base" in model_name
    return gr.update(visible=show_instruction), gr.update(visible=show_speaker), gr.update(visible=show_clone), gr.update(visible=show_clone)

# Gradio Interface
with gr.Blocks(title="Qwen3-TTS Web UI") as demo:
    gr.Markdown("# рџЋ™пёЏ Qwen3-TTS Web UI")
    gr.Markdown("Р“РµРЅРµСЂР°С†РёСЏ СЂРµС‡Рё СЃ РёСЃРїРѕР»СЊР·РѕРІР°РЅРёРµРј `nano-vLLM`. РњРѕРґРµР»СЊ Р·Р°РіСЂСѓР¶Р°РµС‚СЃСЏ РІ РїР°РјСЏС‚СЊ Р°РІС‚РѕРјР°С‚РёС‡РµСЃРєРё РїСЂРё РїРµСЂРІРѕРј Р·Р°РїСЂРѕСЃРµ.")
    
    with gr.Row():
        with gr.Column():
            model_dropdown = gr.Dropdown(
                choices=[
                    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", # РЎРѕР·РґР°РЅРёРµ РіРѕР»РѕСЃР° РїРѕ С‚РµРєСЃС‚Сѓ
                    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice", # Р›РµРіРєР°СЏ РјРѕРґРµР»СЊ СЃ РіРѕС‚РѕРІС‹РјРё РґРёРєС‚РѕСЂР°РјРё
                    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", # РўСЏР¶РµР»Р°СЏ РјРѕРґРµР»СЊ СЃ РіРѕС‚РѕРІС‹РјРё РґРёРєС‚РѕСЂР°РјРё
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",        # Р‘Р°Р·РѕРІР°СЏ РјРѕРґРµР»СЊ РєР»РѕРЅРёСЂРѕРІР°РЅРёСЏ
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base"         # Р‘Р°Р·РѕРІР°СЏ РјРѕРґРµР»СЊ РєР»РѕРЅРёСЂРѕРІР°РЅРёСЏ (РєР°С‡РµСЃС‚РІРµРЅРЅР°СЏ)
                ],
                value="Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                label="Р’С‹Р±РѕСЂ РјРѕРґРµР»Рё"
            )
            
            text_input = gr.Textbox(
                label="РўРµРєСЃС‚ РґР»СЏ РѕР·РІСѓС‡РєРё", 
                lines=5, 
                placeholder="Р’РІРµРґРёС‚Рµ С‚РµРєСЃС‚ Р·РґРµСЃСЊ...",
                value="РџСЂРёРІРµС‚! РЇ С‚РµСЃС‚РёСЂСѓСЋ СЂР°Р±РѕС‚Сѓ РјРѕРґРµР»Рё Qwen3-TTS РЅР° СЂСѓСЃСЃРєРѕРј СЏР·С‹РєРµ С‡РµСЂРµР· РІРµР±-РёРЅС‚РµСЂС„РµР№СЃ."
            )
            
            # This field is shown for VoiceDesign
            instruction_input = gr.Textbox(
                label="РћРїРёСЃР°РЅРёРµ РіРѕР»РѕСЃР° (Instruction)", 
                lines=2, 
                placeholder="РќР°РїСЂРёРјРµСЂ: Р–РµРЅСЃРєРёР№ РіРѕР»РѕСЃ, РїСЂРѕС„РµСЃСЃРёРѕРЅР°Р»СЊРЅС‹Р№ РґРёРєС‚РѕСЂ, С‡РµС‚РєР°СЏ СЂРµС‡СЊ",
                value="Р–РµРЅСЃРєРёР№ РіРѕР»РѕСЃ, СЂР°РґРѕСЃС‚РЅС‹Р№ Рё СЌРЅРµСЂРіРёС‡РЅС‹Р№",
                visible=True
            )
            
            # This field is shown for CustomVoice
            speaker_dropdown = gr.Dropdown(
                label="Р’С‹Р±РѕСЂ РІСЃС‚СЂРѕРµРЅРЅРѕРіРѕ РґРёРєС‚РѕСЂР° (Speaker)",
                choices=['serena', 'vivian', 'uncle_fu', 'ryan', 'aiden', 'ono_anna', 'sohee', 'eric', 'dylan'],
                value="serena",
                visible=False
            )
            
            # These fields are shown for Voice Cloning (Base models)
            ref_audio_input = gr.Audio(
                label="РђСѓРґРёРѕ-РѕСЂРёРіРёРЅР°Р» РґР»СЏ РєР»РѕРЅРёСЂРѕРІР°РЅРёСЏ РіРѕР»РѕСЃР°",
                type="filepath",
                visible=False
            )
            
            ref_text_input = gr.Textbox(
                label="РўРµРєСЃС‚ РёР· РѕСЂРёРіРёРЅР°Р»СЊРЅРѕРіРѕ Р°СѓРґРёРѕ",
                lines=2,
                placeholder="РћСЃС‚Р°РІСЊС‚Рµ РїСѓСЃС‚С‹Рј РґР»СЏ РђР’РўРћРњРђРўРР§Р•РЎРљРћР“Рћ Р РђРЎРџРћР—РќРђР’РђРќРРЇ С‡РµСЂРµР· Whisper",
                visible=False
            )
            
            language_dropdown = gr.Dropdown(
                choices=["Auto", "Russian", "English", "Chinese", "Japanese", "German", "French", "Spanish"],
                value="Russian",
                label="РЇР·С‹Рє"
            )
            
            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                step=0.05,
                value=0.9,
                label="Temperature"
            )
            
            generate_btn = gr.Button("Р“РµРЅРµСЂРёСЂРѕРІР°С‚СЊ рџ”Љ", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Р РµР·СѓР»СЊС‚РёСЂСѓСЋС‰РµРµ Р°СѓРґРёРѕ", type="filepath")
            status_output = gr.Textbox(label="РЎС‚Р°С‚СѓСЃ / Р›РѕРі Р·Р°РіСЂСѓР·РєРё", interactive=False, lines=4)
            
    # Watch for model change to toggle visibility of UI elements
    model_dropdown.change(
        fn=update_ui_for_model,
        inputs=[model_dropdown],
        outputs=[instruction_input, speaker_dropdown, ref_audio_input, ref_text_input]
    )
            
    generate_btn.click(
        fn=generate_tts,
        inputs=[model_dropdown, text_input, instruction_input, speaker_dropdown, ref_audio_input, ref_text_input, language_dropdown, temperature_slider],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Default(primary_hue="blue", secondary_hue="indigo"))

