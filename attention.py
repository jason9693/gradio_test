from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
from torch.nn import functional as F
import seaborn
import matplotlib.pyplot as plt
import io
from PIL import Image

import matplotlib.font_manager as fm

font_path = r'NanumGothicCoding.ttf'
fontprop = fm.FontProperties(fname=font_path, size=18)

plt.rcParams["font.family"] = 'NanumGothic'
hatespeech_category_map = {
    "0": "일반글",
    "1": "공격발언",
    "2": "차별발언"
}


def visualize_attention(sent, attention_matrix, n_words=10):
    def draw(data, x, y, ax):
        seaborn.heatmap(data, 
                        xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
                        cbar=False, ax=ax)
        
    # make plt figure with 1x6 subplots
    fig = plt.figure(figsize=(16, 8))
    # fig.subplots_adjust(hspace=0.7, wspace=0.2)
    for i, layer in enumerate(range(1, 12, 2)):
        ax = fig.add_subplot(2, 3, i+1)
        ax.set_title("Layer {}".format(layer))
        draw(attention_matrix[layer], sent if layer > 6 else [], sent if layer in [1,7] else [], ax=ax)
 
    fig.tight_layout()
    plt.close()
    # fig, axs = plt.subplots(1,6, figsize=(20, 10))

    # for layer in range(1, 12, 2):
    #     print("Encoder Layer", layer+1)
    #     draw(attention_matrix[layer], sent, sent if layer < 2 else [], ax=axs[int(layer/2)])
    
    # plt.show()
    # return fig
        
    # for layer in range(1, 6, 2):
    #     fig, axs = plt.subplots(1,4, figsize=(20, 10))
    #     print("Decoder Self Layer", layer+1)
    #     for h in range(4):
    #         draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
    #             tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
    #     plt.show()
    #     print("Decoder Src Layer", layer+1)
    #     fig, axs = plt.subplots(1,4, figsize=(20, 10))
    #     for h in range(4):
    #         draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
    #             sent, tgt_sent if h ==0 else [], ax=axs[h])
    #     plt.show()

    return fig



def predict(model_name, text):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    tokenized_text = tokenizer([text], return_tensors='pt')
    model.eval()
    output, attention = model(**tokenized_text, output_attentions=True, return_dict=False)
    output = F.softmax(output, dim=-1)
    result = {}
    
    for idx, label in enumerate(output[0].detach().numpy()):
        result[hatespeech_category_map[str(idx)]] = float(label)
    fig = visualize_attention(tokenizer.convert_ids_to_tokens(tokenized_text.input_ids[0]), attention[0][0].detach().numpy())
    return result, fig#.logits.detach()#.numpy()#, output.attentions.detach().numpy()


if __name__ == '__main__':

    model_name = 'jason9693/SoongsilBERT-beep-base'
    text = 'This is a test'
    output = predict(model_name, text)
    
    print(output)


    #Create a gradio app with a button that calls predict()
    app = gr.Interface(fn=predict, server_port=22333, server_name='0.0.0.0', inputs=['text', 'text'], outputs=['label', 'plot'])
    app.launch(inline=False)
