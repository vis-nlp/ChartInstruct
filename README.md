# ChartInstruct: Instruction Tuning for Chart Comprehension and Reasoning

* Authors: [Ahmed Masry](https://ahmedmasryku.github.io/)*, Mehrad Shahmohammadi*, Md Rizwan Parvez, Enamul Hoque, Shafiq Joty (*equal contribution)
* Paper Link: [ChartInstruct](https://arxiv.org/abs/2403.09028)
* Venue: **ACL 2024 Findings**
  
![Screenshot 2024-06-21 215938](https://github.com/vis-nlp/ChartInstruct/assets/47740795/a08ceaa3-39a4-48e2-8064-1a76abc7b2e1)

## ChartInstruct Model Checkpoints
We release the checkpoint for our pretrained model on huggingface. 
| Task  | Checkpoint Path |
| ------------- | ------------- |
| ChartInstruct-Llama2  | [ChartInstruct-Llama2](https://huggingface.co/ahmed-masry/ChartInstruct-LLama2)  |
| ChartInstruct-Flan-T5-XL  | Coming soon |

**IMPORTANT:** Please note that we have changed the alignment module from a linear layer (as described in the paper) to an MLP with 2 layers to improve the compatability with huggignface's LLaVA codebase. This made our models very easy to run and finetune using a few lines of code as you will see below!

## Web Demo
If you wish to quickly try our models, you can access our public web demoes hosted on the Hugging Face Spaces platform with a friendly interface!

| Tasks  | Web Demo |
| ------------- | ------------- |
| ChartInstruct-Llama2  | [ChartInstruct-Llama2](https://huggingface.co/spaces/ahmed-masry/ChartInstruct-LLama2) |
| ChartInstruct-Flan-T5-XL  | Coming soon |

## Inference
You can easily use our models for inference with the huggingface library! You just need to do the following:

Chage the image_path to your chart example image path on your system
Write the input_text
We recommend using **beam search** with a beam size of 4 to better results, but if your machine's GPU has low memory, you can remove the num_beams from the generate method.


```
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

image_path = "/content/chart_example_1.png"
input_text = "What is the share of respondants who prefer Whatsapp in the 18-29 age group?"

input_prompt = f"<image>\n Question: {input_text} Answer: "

model = LlavaForConditionalGeneration.from_pretrained("ahmed-masry/ChartInstruct-LLama2", torch_dtype=torch.float16)
processor = AutoProcessor.from_pretrained("ahmed-masry/ChartInstruct-LLama2")


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

image = Image.open(image_path).convert('RGB')
inputs = processor(text=input_prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# change type if pixel_values in inputs to fp16. 
inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
prompt_length = inputs['input_ids'].shape[1]

# move to device
inputs = {k: v.to(device) for k, v in inputs.items()}

# Generate
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)

```

Does you GPU have low memory? The above code is slow on your machine? **We got you covered!** Use the following code that loads thq **quantized** version of the model. 
Just make sure to install the following pip modules: bitsandbytes, itsandbytes-cuda112, accelerate

```
from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import torch
from PIL import Image

torch.hub.download_url_to_file('https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/val/png/multi_col_1229.png', 'chart_example_1.png')

image_path = "/content/chart_example_1.png"
input_text = "What is the share of respondants who prefer Whatsapp in the 18-29 age group?"

input_prompt = f"<image>\n Question: {input_text} Answer: "

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
)

model = LlavaForConditionalGeneration.from_pretrained("ahmed-masry/ChartInstruct-LLama2", torch_dtype=torch.float16, quantization_config=bnb_config)
processor = AutoProcessor.from_pretrained("ahmed-masry/ChartInstruct-LLama2")

image = Image.open(image_path).convert('RGB')

inputs = processor(text=input_prompt, images=image, return_tensors="pt")

# change type if pixel_values in inputs to fp16. 
inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)
prompt_length = inputs['input_ids'].shape[1]


# Generate
generate_ids = model.generate(**inputs, num_beams=4, max_new_tokens=512)
output_text = processor.batch_decode(generate_ids[:, prompt_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output_text)
```

## Finetuning 
Checkout the example colab notebook in the repo that shows how to finetune the model on the ChartQA Dataset. 
The training code is optimized such that you can train it on a **T4 GPU** which is **free on Colab**. 
The notebook has three different setups LoRA & QLoRA & Full Finetuning. Based on your machine's GPU, you can switch between them. 
This notebook was adapted from [NielsRogge Tutorials](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/LLaVa/Fine_tune_LLaVa_on_a_custom_dataset_(with_PyTorch_Lightning).ipynb)

# Contact
If you have any questions about this work, please contact **[Ahmed Masry](https://ahmedmasryku.github.io/)** using the following email addresses: **amasry17@ku.edu.tr** or **ahmed.elmasry24653@gmail.com**.

# Reference
Please cite our paper if you use our models in your research. 

```
@misc{masry2024chartinstruct,
      title={ChartInstruct: Instruction Tuning for Chart Comprehension and Reasoning}, 
      author={Ahmed Masry and Mehrad Shahmohammadi and Md Rizwan Parvez and Enamul Hoque and Shafiq Joty},
      year={2024},
      eprint={2403.09028},
      archivePrefix={arXiv},
      primaryClass={id='cs.CL' full_name='Computation and Language' is_active=True alt_name='cmp-lg' in_archive='cs' is_general=False description='Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'}
}
```
