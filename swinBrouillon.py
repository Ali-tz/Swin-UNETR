
# https://huggingface.co/docs/transformers/model_doc/swin 
# class transformers.swinConfig
# ( image_size = 224,
# patch_size = 4,
# num_channels = 3,
# embed_dim = 96,
# depths = [2, 2, 6, 2],
# num_heads = [3, 6, 12, 24],
# window_size = 7,
# mlp_ratio = 4.0,
# qkv_bias = True,
# hidden_dropout_prob = 0.0,
# attention_probs_dropout_prob = 0.0,
# drop_path_rate = 0.1,
# hidden_act = 'gelu',
# use_absolute_embeddings = False,
# initializer_range = 0.02,
# layer_norm_eps = 1e-05,
# encoder_stride = 32,
# out_features = None,
#  out_indices = None**kwargs )

### SwinModel

from transformers import SwinConfig, SwinModel

# Initializing a Swin microsoft/swin-tiny-patch4-window7-224 style configuration
configuration = SwinConfig()

# Initializing a model (with random weights) from the microsoft/swin-tiny-patch4-window7-224 style configuration
model = SwinModel(configuration)

# Accessing the model configuration
configuration = model.config


### SwinForMaskedImageModeling
# class transformers.swinModel
# ( configadd_pooling_layer = Trueuse_mask_token = False )

# ( pixel_values: typing.Optional[torch.FloatTensor] = Nonebool_masked_pos: typing.Optional[torch.BoolTensor] = Nonehead_mask: typing.Optional[torch.FloatTensor] = Noneoutput_attentions: typing.Optional[bool] = Noneoutput_hidden_states: typing.Optional[bool] = Nonereturn_dict: typing.Optional[bool] = None ) → transformers.models.swin.modeling_swin.SwinModelOutput or tuple(torch.FloatTensor)

from transformers import AutoImageProcessor, SwinModel
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)



from transformers import AutoImageProcessor, SwinForMaskedImageModeling
import torch
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
model = SwinForMaskedImageModeling.from_pretrained("microsoft/swin-base-simmim-window6-192")

num_patches = (model.config.image_size // model.config.patch_size) ** 2
pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
# create random boolean mask of shape (batch_size, num_patches)
bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
list(reconstructed_pixel_values.shape)


### SwinForImageClassification
from transformers import AutoImageProcessor, SwinForImageClassification
import torch
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
print(model.config.id2label[predicted_label])

# ## TFSwinModel
from transformers import AutoImageProcessor, TFSwinModel
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = TFSwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

inputs = image_processor(image, return_tensors="tf")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
list(last_hidden_states.shape)

### TFSwinForMaskedImageModeling

from transformers import AutoImageProcessor, TFSwinForMaskedImageModeling
import tensorflow as tf
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = TFSwinForMaskedImageModeling.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

num_patches = (model.config.image_size // model.config.patch_size) ** 2
pixel_values = image_processor(images=image, return_tensors="tf").pixel_values
# create random boolean mask of shape (batch_size, num_patches)
bool_masked_pos = tf.random.uniform((1, num_patches)) >= 0.5

outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
list(reconstructed_pixel_values.shape)


#### TFSwinForImageClassification

from transformers import AutoImageProcessor, TFSwinForImageClassification
import tensorflow as tf
from datasets import load_dataset

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
model = TFSwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

inputs = image_processor(image, return_tensors="tf")
logits = model(**inputs).logits

# model predicts one of the 1000 ImageNet classes
predicted_label = int(tf.math.argmax(logits, axis=-1))
print(model.config.id2label[predicted_label])