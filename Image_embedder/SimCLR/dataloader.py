import torch


class TensorData():

    def __init__(self, x_data, y_data):

        self.x_data = torch.FloatTensor(x_data)
        self.y_data = torch.LongTensor(y_data)

        self.len = self.y_data.shape[0]
    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len