import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Classifier(nn.Module):
	def __init__(self):
		super(CNN_Classifier, self).__init__()
		self.cnn1 = nn.Sequential(
				nn.Conv1d(1, 64, kernel_size=50, stride=6, padding=1, bias=False),
				nn.BatchNorm1d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.MaxPool1d(kernel_size=8, stride=8),
				nn.Dropout(0.5),

				nn.Conv1d(64, 128, kernel_size=8, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.MaxPool1d(kernel_size=4, stride=4)
			)
		self.cnn2 = nn.Sequential(
				nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=1, bias=False),
				nn.BatchNorm1d(64),
				nn.LeakyReLU(0.2, inplace=True),
				nn.MaxPool1d(kernel_size=4, stride=4),
				nn.Dropout(0.5),
				
				nn.Conv1d(64, 128, kernel_size=6, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.Conv1d(128, 128, kernel_size=6, stride=1, padding=1, bias=False),
				nn.BatchNorm1d(128),
				nn.LeakyReLU(0.2, inplace=True),

				nn.MaxPool1d(kernel_size=2, stride=2)
			)


		self.dnn = nn.Sequential(
			nn.Dropout(0.5),
			nn.Linear(6912, 512),
			nn.SELU(),
			nn.Dropout(0.5),
			nn.Linear(512, 1024),
			nn.SELU(),
			nn.Dropout(0.5),
			nn.Linear(1024, 512),
			nn.SELU(),
			nn.Dropout(0.5),
			nn.Linear(512,2)
		)

	def forward(self, x):
		# x shape (batch, time_step, input_size)
		x = x.permute(0,2,1)
		out1 = self.cnn1(x).view(len(x),-1)
		out2 = self.cnn2(x).view(len(x),-1)
		dnn_in = torch.cat((out1,out2), 1)
		output = self.dnn(dnn_in)
		return output

class Balence_Net_Classifier(nn.Module):
	def __init__(self):
		super(Balence_Net_Classifier, self).__init__()

		self.lstm1 = nn.LSTM(input_size=1, hidden_size=512, num_layers=2,batch_first=True,dropout=0.5, bidirectional=True)
		self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=2,batch_first=True,dropout=0.5, bidirectional=True)

		self.r2c1 = nn.Sequential(
		nn.Conv1d(1024,256, kernel_size=2, stride=1),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5))
		self.r2c2 = nn.Sequential(
		nn.Conv1d(1024,256, kernel_size=3, stride=1),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5))
		self.r2c3 = nn.Sequential(
		nn.Conv1d(1024,256, kernel_size=5, stride=1),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5))
		self.r2c4 = nn.Sequential(
		nn.Conv1d(1024,256, kernel_size=6, stride=1),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5))
		self.r2c5 = nn.Sequential(
		nn.Conv1d(1024,256, kernel_size=8, stride=1),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5))

		self.c2r1 = nn.Sequential(
		nn.Conv1d(1,256, kernel_size=2, stride=1),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5))
		self.c2r2 = nn.Sequential(
		nn.Conv1d(1,256, kernel_size=3, stride=1),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5))
		self.c2r3 = nn.Sequential(
		nn.Conv1d(1,256, kernel_size=4, stride=1),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5))

		self.maxpool = nn.Sequential(
		nn.MaxPool1d(kernel_size=4, stride=4),
		nn.Dropout(0.5),
		)

		self.dnn = nn.Sequential(
		nn.Linear(18688,1024),
		nn.ReLU(),
		nn.BatchNorm1d(1024),
		nn.Dropout(0.5),	
		nn.Linear(1024,256),
		nn.ReLU(),
		nn.BatchNorm1d(256),
		nn.Dropout(0.5),
		nn.Linear(256,2)
		)


	def forward(self, x):
		# x shape (batch, time_step, input_size)
		# r_out shape (batch, time_step, output_size)
		# h_n shape (n_layers, batch, hidden_size)
		# h_c shape (n_layers, batch, hidden_size)
		r_out1, (hn1,cn1) = self.lstm1(x, None)
		r_out1 = r_out1.permute(0,2,1)
		cat1 = torch.cat((self.r2c1(r_out1),self.r2c2(r_out1),self.r2c3(r_out1),self.r2c4(r_out1),self.r2c5(r_out1)),2)
		
		c2r_out1 = self.c2r1(x.permute(0,2,1))
		c2r_out2 = self.c2r2(x.permute(0,2,1))
		c2r_out3 = self.c2r3(x.permute(0,2,1))
		cat2 = torch.cat((c2r_out1,c2r_out2,c2r_out3),2)
		cat2 = cat2.permute(0,2,1)
		#print(cat2.shape)
		#input()
		r_out2, (hn2,cn2) = self.lstm2(cat2, None)
		r_out2 = r_out2.permute(0,2,1)

		cat3 = torch.cat((cat1,r_out2),2)
		cat3 = self.maxpool(cat3).view(x.size()[0],-1)
		y = self.dnn(cat3)
		return y