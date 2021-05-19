import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
## data_loader
def get_data_loader(data_list, batch_size, shuffle=False,num_workers=10,  drop_last=True):
    dataset = Make_Dataset(data_list)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, collate_fn=make_padding)
    return data_loader

def make_padding(samples):
    def padd(samples):
        length = [len(s) for s in samples]
        max_length = max(length)
        batch = torch.zeros(len(length), max_length).to(torch.long)
        for idx, sample in enumerate(samples):
            batch[idx, :length[idx]] = torch.LongTensor(sample)
        return torch.LongTensor(batch)
    encoder = [sample["encoder"] for sample in samples]
    decoder = [sample["decoder"] for sample in samples]
    encoder = padd(encoder)
    decoder = padd(decoder)

    return {'encoder':encoder.contiguous(),"decoder":decoder.contiguous()}


class Make_Dataset(Dataset):
    def __init__(self, path):
        self.encoder_input = torch.load(path[0])
        self.decoder_input = torch.load(path[1])

        self.encoder_input = np.array(self.encoder_input, dtype=object)
        self.decoder_input = np.array(self.decoder_input, dtype=object)

    def __len__(self):
        return len(self.encoder_input)

    def __getitem__(self, idx):
        return {"encoder":torch.LongTensor(self.encoder_input[idx]), "decoder":torch.LongTensor(self.decoder_input[idx])}


# path = ["/hdd/user15/workspace/Transformer/data/prepro/en_de/test.en.txt","/hdd/user15/workspace/Transformer/data/prepro/en_de/test.de.txt"]
# dataloader= get_data_loader(path, 10)
# len(dataloader)
# for i in dataloader:
#     print(i)










