import os
import sys

glm4voice_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'models', 'glm4voice'))
sys.path.append(glm4voice_path)

glm4voice_third_party_path = os.path.abspath(
    os.path.join(glm4voice_path, 'third_party/Matcha-TTS'))
sys.path.append(glm4voice_third_party_path)

stepaudio_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'models', 'stepaudio'))
sys.path.append(stepaudio_path)