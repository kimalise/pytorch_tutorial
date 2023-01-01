import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False) -> None:
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        self.relu = nn.Relu()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)

        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN
        
        return self.dropout(self.relu(features))

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_sized, hidden_size, num_layers) -> None:
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_sized)
        self.lstm = nn.LSTM(embed_sized, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        '''
            features: [batch_size, feature_dim]
            captions: [seq_len, batch_size]
        '''
        embeddings = self.dropout(self.embed(captions)) # [seq_len, batch_size, embed_size]
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0) # [seq_len + 1, batch_size, embed_size]
        outputs, _ = self.lstm(embeddings)
        return outputs

class CNNtoRNN(nn.Module): 
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers) -> None:
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(vocab_size, embed_size, hidden_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0) # [1, batch_size, embed_size]
            state = None

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, state)
                output = self.decoder.linear(hiddens.unsqueeze(0))
                predicted = output.argmax(1)

                result_caption.append(predicted.item())
                x = self.decoder.embed(predicted).unsqueeze(0)
                
                if vocabulary.itos[predicted.item()] == '<EOS>':
                    break

        return [vocabulary.itos[idx] for idx in result_caption]


if __name__ == "__main__":
    pass






