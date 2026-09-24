from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch

def get_hidden_data(model_path):
    # load model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    # model = model.cuda()

    # load target text
    total_title = []
    total_target = []
    with open('./xsum/test.source') as source:
        total_title = source.readlines()
        for i in range(len(total_title)):
            total_title[i] = total_title[i].strip()
    with open('./xsum/test.target') as target:
        total_target = target.readlines()
        for i in range(len(total_target)):
            total_target[i] = total_target[i].strip()

    # initialize saved data
    encoder = []
    decoder = []
    with torch.no_grad():
        for i in range(len(total_title)):
            if 't5' in model_path:
                input_ids = tokenizer(
                    "summarize: " + total_title[i], return_tensors="pt"
                ).input_ids  # Batch size 1
            else:
                input_ids = tokenizer(
                    total_title[i], return_tensors="pt"
                ).input_ids  # Batch size 1
            # generate summary
            outputs = model.generate(input_ids=input_ids,
                                     output_hidden_states=True,
                                     return_dict_in_generate=True,
                                     max_length=64,  # +2 from original because we start at step=1 and stop before max_length
                                     # min_length=5,  # +1 from original because we start at step=1
                                     no_repeat_ngram_size=3,
                                     num_beams=8,
                                     length_penalty=0.6,
                                     early_stopping=True,)
            # extraction data from all outputs
            encoder_hidden_states = outputs['encoder_hidden_states']
            decoder_hidden_states = outputs['decoder_hidden_states']
            if encoder == []:
                for j in range(encoder_hidden_states.shape[0]):
                    encoder.append(encoder_hidden_states[j][0])
                    decoder.append(decoder_hidden_states[j][0])
            else:
                for j in range(encoder_hidden_states.shape[0]):
                    encoder[j] = torch.cat((encoder[j],encoder_hidden_states[j][0]), dim=0)
                    decoder[j] = torch.cat((decoder[j],decoder_hidden_states[j][0]), dim=0)
    # save data
    np.save(f'{model_path}/encoder_hidden_states.npy',torch.tensor(encoder))
    np.save(f'{model_path}/decoder_hidden_states.npy',torch.tensor(decoder))