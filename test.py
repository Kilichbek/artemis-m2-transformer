import random
from data import ArtEmisDetectionsField, TextField, RawField, EmotionField
from data import ArtEmis, DataLoader
import evaluation
from models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory
import torch
from torch.nn import functional as F
from tqdm import tqdm
import argparse
import pickle
import numpy as np

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)

emotions_dict = { 
    'amusement': 0 , 
    'awe' : 1, 
    'contentment' : 2, 
    'excitement' : 3, 
    'anger' : 4, 
    'disgust' : 5, 
    'fear' : 6, 
    'sadness': 7, 
    'something else' : 8
}
def predict_captions(model, emotion_encoder, dataloader, text_field):
    import itertools
    emotion_encoder.eval()
    model.eval()
    gen = {}
    gts = {}
    with tqdm(desc='Evaluation', unit='it', total=len(dataloader)) as pbar:
        
        for it, (images, emotions_caps_pair) in enumerate(iter(dataloader)):
            caps_gt, emotions = emotions_caps_pair

            images = images.to(device)

            emotions = torch.stack([torch.mode(emotion).values for emotion in emotions])
            emotions = F.one_hot(emotions, num_classes=9)
            emotions = emotions.type(torch.FloatTensor)
            emotions = emotions.to(device)
            enc_emotions = emotion_encoder(emotions)
            enc_emotions = enc_emotions.unsqueeze(1).repeat(1, images.shape[1], 1)
            images = torch.cat([images, enc_emotions], dim=-1)

            with torch.no_grad():
                out, _ = model.beam_search(images, 20, text_field.vocab.stoi['<eos>'], 5, out_size=1)
            caps_gen = text_field.decode(out, join_words=False)
            for i, (gts_i, gen_i) in enumerate(zip(caps_gt, caps_gen)):
                gen_i = ' '.join([k for k, g in itertools.groupby(gen_i)])
                gen['%d_%d' % (it, i)] = [gen_i.strip(), ]
                gts['%d_%d' % (it, i)] = gts_i
            pbar.update()

    store_dict = {'gen': gen,'gts': gts} 
    with open('test_class.pickle', 'wb') as f:
        pickle.dump(store_dict,f)

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gen = evaluation.PTBTokenizer.tokenize(gen)
    scores, _ = evaluation.compute_scores(gts, gen)

    return scores


if __name__ == '__main__':
    device = torch.device('cuda')

    parser = argparse.ArgumentParser(description='Meshed-Memory Transformer')

    parser.add_argument('--exp_name', type=str, default='m2_transformer')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--features_path', type=str)
    parser.add_argument('--annotation_folder', type=str)
    args = parser.parse_args()

    print('Meshed-Memory Transformer Evaluation')

    # Pipeline for image regions
    image_field = ArtEmisDetectionsField(detections_path=args.features_path, max_detections=50, load_in_tmp=False)

    # Pipeline for text
    text_field = TextField(init_token='<bos>', eos_token='<eos>', lower=True, tokenize='spacy',
                           remove_punctuation=True, nopoints=False)

    # Pipeline for emotion
    emotions = [
        'amusement', 'awe', 'contentment', 'excitement', 
        'anger', 'disgust', 'fear', 'sadness', 'something else'
        ]
    emotion_field = EmotionField(emotions=emotions)

    # Create the dataset
    path_to_images = '/wiki_art_paintings/rescaled_600px_max_side/'
    dataset = ArtEmis(image_field, text_field, emotion_field, path_to_images, args.annotation_folder)
    _, _, test_dataset = dataset.splits
    text_field.vocab = pickle.load(open('vocab_%s.pkl' % args.exp_name, 'rb'))

    # Model and dataloaders
    encoder = MemoryAugmentedEncoder(3, 0, attention_module=ScaledDotProductAttentionMemory,
                                     attention_module_kwargs={'m': 40}, d_in=2058)
    decoder = MeshedDecoder(len(text_field.vocab), 54, 3, text_field.vocab.stoi['<pad>'])
    model = Transformer(text_field.vocab.stoi['<bos>'], encoder, decoder).to(device)
    emotion_encoder = torch.nn.Sequential(
        torch.nn.Linear(9,10)
    )
    emotion_encoder.to(device)

    fname = 'saved_models/%s_best.pth' % args.exp_name
    data = torch.load(fname)
    model.load_state_dict(data['state_dict'])

    fname = 'saved_models/%s_emotion_best.pth' % args.exp_name
    data = torch.load(fname)
    emotion_encoder.load_state_dict(data)

    dict_dataset_test = test_dataset.image_dictionary({'image': image_field, 'text': RawField(), 'emotion': emotion_field})
    dict_dataloader_test = DataLoader(dict_dataset_test, batch_size=args.batch_size, num_workers=args.workers)

    scores = predict_captions(model, emotion_encoder, dict_dataloader_test, text_field)
    print(scores)
