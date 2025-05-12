import soundfile as sf

from dia.model import Dia
import torch


# model = Dia.from_pretrained("nari-labs/Dia-1.6B")
model = Dia.from_local(checkpoint_path="../checkpoints-rusisan/ckpt_epoch1.pth", config_path="../dia/config.json")
# model = Dia.from_local(checkpoint_path="../checkpoints/ckpt_epoch8.pth", config_path="../dia/config.json")
model.model.to(dtype=torch.float32)

text = ("[S1] Привет, я надеюсь я здесь. [S2]То есть нужно оценить примеры текстов копирайтинга, которые он писал для "
        "других сайтов. "
        "[S1]Но как оценивать копирайтеров по примерам статей?  (laughs) ")

output = model.generate(text)

sf.write("simple-ru-v1.mp3", output, 44100)
