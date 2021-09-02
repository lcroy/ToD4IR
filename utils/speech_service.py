import azure.cognitiveservices.speech as speechsdk
import time

# Microsoft service to recognize speech
def speech_to_text_microsoft():
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False
    result = ''

    def stop_cb(evt):
        """callback that stops continuous recognition upon receiving an event `evt`"""
        nonlocal result
        result = evt.result.text
        # Stop continuous speech recognition
        speech_recognizer.stop_continuous_recognition()
        nonlocal done
        done = True
        print("User: " + result)

        # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(lambda evt: stop_cb(evt) if evt.result.text != "" else print("Max: I did not hear anything..."))

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.1)

    return result.lower()


def text_to_speech_microsoft(text):
    # Creates an instance of a speech config with specified subscription key and service region.
    # Replace with your own subscription key and service region (e.g., "westus").
    speech_key, service_region = "f6bd8f851e48430ea0ea46bb47fad10a", "northeurope"
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
    speech_config.speech_synthesis_voice_name = "en-GB-RyanNeural"

    # Creates a speech synthesizer using the default speaker as audio output.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    print("Max: " + text)
    result = speech_synthesizer.speak_text(text)


speech_to_text_microsoft()
