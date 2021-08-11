from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr
from torch.nn import functional as F
import seaborn
import matplotlib.pyplot as plt
import io
from PIL import Image

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
    
    def plot_to_image(figure=None):
        if figure is None:
            figure = plt.gcf()
        buf = io.BytesIO()
        figure.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        plt.close(figure)
        return image
        
    # make plt figure with 1x6 subplots
    fig = plt.figure()
    # fig.subplots_adjust(hspace=0.7, wspace=0.2)
    for i, layer in enumerate(range(1, 12, 2)):
        ax = fig.add_subplot(2, 3, i+1)
        draw(attention_matrix[layer], sent, sent if layer < 2 else [], ax=ax)
        
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

    return plot_to_image(fig)



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
    
    # label = torch.argmax(output[0], dim=-1).numpy()
    #attn = output[1]
    # print(output[0].size())
    #print(attn)
    print(output)


    #Create a gradio app with a button that calls predict()
    app = gr.Interface(fn=predict, inputs=['text', 'text'], outputs=['label', 'image'])
    app.launch(inline=False)
